# Transport Engine Rewrite Implementation Guide

## Full Pre-Implementation Guidance for Replacing Kwant with Native RGF

This document is the implementation guide for replacing the current Kwant
transport path with a native RGF transport engine inside `wtec`.

It is not a benchmark memo and not a speculative note. It is the working
guidance that should be followed before code is written.

This guide assumes the runtime contract is fixed:

- all heavy runs are submitted through `qsub`
- all heavy runs are launched under `mpirun`
- fork launchstyle is forbidden
- queue families are `g4`, `g3`, `g2`, `g1`

This guide also assumes the package remains element agnostic:

- transport sizing comes from the actual Wannier HR and geometry
- no element-specific solver branches
- no hardcoded orbital counts or principal-layer widths by material name

Companion document:

- [rgf_engine_design_spec.md](/home/msj/Desktop/playground/electroics/wtec/docs/rgf_engine_design_spec.md)

The design spec defines the target architecture. This document defines how to
implement it safely in the repo and on the cluster.

---

## 1. Rewrite Objective

The rewrite target is narrow and explicit:

- replace the transport hot path currently implemented through
  [conductance.py](/home/msj/Desktop/playground/electroics/wtec/wtec/transport/conductance.py)
- keep topology, DFT, Wannier generation, and cluster infrastructure intact
- keep transport payload/result semantics stable so the rest of `wtec` does not
  need to be rewritten at the same time

The intended transport backend is:

- native C
- MPI executable
- launched directly under `mpirun`
- submitted through the existing PBS/qsub flow

The rewrite is not complete until:

1. `wtec init` can prepare the cluster-side router automatically
2. `wtec run` can submit a real transport job through `qsub`
3. the job runs through `mpirun`
4. the result matches the current Kwant path on validation cases

---

## 2. Hard Constraints

These constraints are non-negotiable. The implementation must respect them
from the first commit.

### 2.1 Submission and Launch

- submission is always `qsub`
- compute launch is always `mpirun`
- no fork launchstyle
- no Python process is allowed to be the transport compute engine under MPI

That means the final transport command must look like:

```bash
mpirun -np <mpi_np> ./wtec_rgf_runner transport_payload.json transport_result.json
```

Not allowed:

- `python -m ...` as the actual transport compute engine
- `mpi4py`
- `ctypes`
- `multiprocessing`
- any fork-based worker pool

### 2.2 Cluster Fit

The current queue model must remain the source of truth:

- `g4`
- `g3`
- `g2`
- `g1`

Per-queue cores are already resolved in:

- [cluster.py](/home/msj/Desktop/playground/electroics/wtec/wtec/config/cluster.py)

PBS resources are already emitted by:

- [pbs.py](/home/msj/Desktop/playground/electroics/wtec/wtec/cluster/pbs.py)

The new backend must integrate with those files, not invent a separate queue
layer.

### 2.3 Physics Scope

Phase 1 production support is:

- clean ordered transport
- exact periodic transverse reduction

Phase 1 does not promise:

- arbitrary full finite transport for every slab width
- random disorder along the periodic reduced axis
- silent conversion of a non-periodic problem into a periodic one

### 2.4 Element Agnosticism

The engine must never depend on:

- element name
- material family
- pseudopotential choice
- fixed orbital count assumptions

All sizing must come from:

- `n_orb` in the HR
- actual surviving hopping range in the chosen geometry
- chosen mode
- chosen queue

---

## 3. Current Code That Must Remain Stable

These package areas are not the rewrite target and should only be touched when
integration requires it.

### 3.1 Keep Stable

- [topology](/home/msj/Desktop/playground/electroics/wtec/wtec/topology)
- [qe](/home/msj/Desktop/playground/electroics/wtec/wtec/qe)
- [siesta](/home/msj/Desktop/playground/electroics/wtec/wtec/siesta)
- [wannier](/home/msj/Desktop/playground/electroics/wtec/wtec/wannier) except for
  interface reuse
- current cluster SSH / PBS utilities

### 3.2 Current Files the Rewrite Must Integrate With

- [conductance.py](/home/msj/Desktop/playground/electroics/wtec/wtec/transport/conductance.py)
- [model.py](/home/msj/Desktop/playground/electroics/wtec/wtec/wannier/model.py)
- [orchestrator.py](/home/msj/Desktop/playground/electroics/wtec/wtec/workflow/orchestrator.py)
- [cli.py](/home/msj/Desktop/playground/electroics/wtec/wtec/cli.py)
- [cluster.py](/home/msj/Desktop/playground/electroics/wtec/wtec/config/cluster.py)
- [pbs.py](/home/msj/Desktop/playground/electroics/wtec/wtec/cluster/pbs.py)

### 3.3 Do Not Delete Kwant Early

Kwant remains the validation oracle until the RGF backend has numerical parity.

The initial rewrite must therefore:

- add `rgf`
- keep `kwant`
- only later consider `auto`
- only after validation consider changing defaults

---

## 4. Final Runtime Shape

The package should end up with this runtime flow:

```
wtec init
  ├─ validates cluster connection
  ├─ resolves queue map
  ├─ builds or verifies native RGF binary on cluster
  ├─ writes init manifest / router state
  └─ records RGF availability in .wtec/init_state.json

wtec run
  ├─ builds transport payload
  ├─ parses HR metadata locally for preflight
  ├─ computes safe mpi_np for selected queue
  ├─ stages payload + queue-safe binary
  ├─ writes PBS script
  ├─ qsub submission
  ├─ mpirun launch on cluster
  ├─ retrieves result json + log
  └─ merges result into existing report flow
```

This means:

- `wtec init` is responsible for router readiness
- `wtec run` is responsible for job-specific sizing

---

## 5. Phase Plan

The rewrite must be executed in phases. Do not collapse them.

### Phase 0: Interface Freeze

Goal:

- define transport engine interfaces before solver code exists

Required changes:

- add transport engine enum:
  - `"kwant"`
  - `"rgf"`
  - `"auto"`
- define shared payload keys
- define shared result keys
- define runtime certificate fields

Required runtime certificate fields:

- `engine`
- `queue`
- `mpi_size`
- `threads`
- `mode`
- `n_orb`
- `principal_layer_width`
- `n_super`
- `per_rank_bytes`
- `safe_rank_cap`
- `binary_id`

Deliverable:

- Python schema and config changes only

Gate:

- package can build and validate transport config without any RGF binary

### Phase 1: Native Project Scaffold

Goal:

- create the standalone C MPI transport project

Required repo path:

`wtec/ext/rgf/`

Required files:

- `include/wtec_rgf.h`
- `include/wtec_rgf_internal.h`
- `src/rgf_runner.c`
- `src/rgf_model.c`
- `src/rgf_hamiltonian.c`
- `src/rgf_sancho.c`
- `src/rgf_sweep.c`
- `src/rgf_periodic.c`
- `src/rgf_disorder.c`
- `src/rgf_payload.c`
- `src/rgf_result.c`
- `src/rgf_linalg.c`
- `src/rgf_rng.c`
- `vendor/cJSON.c`
- `vendor/cJSON.h`
- `CMakeLists.txt`
- `build_on_cluster.sh`
- `tests/test_chain.c`
- `tests/test_mpi_4rank.c`

Deliverable:

- `wtec_rgf_runner` builds locally and on the cluster

Gate:

- binary exists and runs `--help` or dummy payload parse successfully

### Phase 2: Router Ownership in `wtec init`

Goal:

- make `wtec init` fully prepare the RGF backend

Required behavior:

- detect cluster compiler, MPI, BLAS/LAPACK environment
- build or verify RGF binary on the cluster
- write cluster-side build manifest
- write local init-state router record

Required output in `.wtec/init_state.json`:

```json
{
  "rgf": {
    "enabled": true,
    "binaries": {
      "generic": {
        "path": "/remote/path/wtec_rgf_runner",
        "isa": "x86-64-v2"
      }
    }
  }
}
```

Important rule:

- the default build must be queue-safe
- do not default to `-march=native`
- do not assume AVX-512 everywhere

Gate:

- `wtec init` alone is sufficient to make later RGF submission possible

### Phase 3: HR Reader and Geometry Sizing

Goal:

- make the solver independent of element-specific logic

Required behavior:

- parse `*_hr.dat`
- parse lattice vectors from `.win` if not explicitly provided
- read `n_orb`
- store hopping lattice vectors and matrices
- compute effective principal-layer width from actual HR and requested geometry

This phase is where the implementation must mirror the existing geometry logic
conceptually from:

- [model.py](/home/msj/Desktop/playground/electroics/wtec/wtec/wannier/model.py)

But the C implementation must not depend on Python object layout.

Gate:

- C code reports the same effective lead-axis span as the Python reference on
  validation cases

### Phase 4: Core Solver

Goal:

- implement the numerical RGF engine

Required subparts:

- superslice Hamiltonian construction
- Sancho-Rubio lead self-energies
- recursive sweep
- Fisher-Lee transmission
- deterministic disorder injection

The Hamiltonian must be built at the superslice level using:

- `P_eff`
- geometry
- mode reduction state

Gate:

- toy nearest-neighbor chain gives analytic conductance
- principal-layer regrouping invariance test passes

### Phase 5: `periodic_transverse` Production Mode

Goal:

- implement the only phase-1 production mode

Required behavior:

- exact Fourier reduction of one transverse axis
- conductance accumulated over the `k_perp` grid
- MPI work is distributed over `k_perp` sectors and outer work items

Critical rule:

- do not loop every `k_perp` sector serially inside one rank by default

Work unit shape:

- `(thickness, disorder_seed, k_index)`

Gate:

- clean periodic cases match current Kwant periodic mode numerically
- node utilization is materially improved vs single-rank serial `k` looping

### Phase 6: Python Orchestrator Integration

Goal:

- make the new backend invocable through existing `wtec` commands

Required behavior:

- transport engine dispatch from Python
- payload writing
- binary staging
- PBS writing
- result retrieval
- report merge

Required PBS shape:

```bash
#PBS -l select=1:ncpus=<queue_cores>:mpiprocs=<mpi_np>:ompthreads=1
mpirun -np <mpi_np> ./wtec_rgf_runner transport_payload.json transport_result.json
```

No Python compute wrapper is allowed here.

Gate:

- a real qsub run completes end-to-end on the cluster

### Phase 7: Memory Fit and Safe Rank Clipping

Goal:

- stop unsafe submissions before they waste queue time or crash nodes

Required behavior:

- parse actual HR metadata in preflight
- compute:
  - `P_eff`
  - `N_super`
  - `per_rank_bytes`
  - `safe_rank_cap`
- choose:
  - `mpi_np = min(queue_cores, n_work_units, safe_rank_cap)`

Required duplicate check:

- the same fit logic must exist in the C executable at startup

Gate:

- oversize jobs fail clearly before or at startup with an actionable message

### Phase 8: Validation and Controlled Cutover

Goal:

- prove the new backend before any default switch

Required validation classes:

1. analytic toy chains
2. small multi-orbital toy systems
3. one real non-SOC HR
4. one real SOC HR

Comparison target:

- current Kwant backend on the same payload and geometry

Acceptance:

- clean relative error `< 1e-4`
- disorder-averaged mean/std within agreed statistical tolerance

Only after this phase:

- add `"auto"`
- consider making `periodic_transverse` default to RGF

### Phase 9: Experimental `full_finite`

Goal:

- extend RGF beyond the phase-1 clean periodic path

This phase must remain opt-in.

Required behavior:

- explicit config mode:
  - `"full_finite"`
- safe rank clipping
- no silent reduction to a periodic mode

Gate:

- correctness established on small finite systems
- memory-fit behavior proven on real HR models

---

## 6. File-by-File Implementation Map

### 6.1 New Native Code

Add under:

- [wtec/ext/rgf](/home/msj/Desktop/playground/electroics/wtec/wtec/ext/rgf)

Responsibility map:

- `rgf_runner.c`
  - `main()`
  - MPI init/finalize
  - payload parse
  - rank distribution
  - runtime certificate
- `rgf_model.c`
  - HR parser
  - lattice parser
  - `n_orb`
  - effective span helpers
- `rgf_hamiltonian.c`
  - superslice blocks
  - mode-specific block construction
- `rgf_sancho.c`
  - lead surface Green function
- `rgf_sweep.c`
  - RGF sweep and Fisher-Lee transmission
- `rgf_periodic.c`
  - periodic axis Fourier reduction
  - `k_perp` work decomposition helpers
- `rgf_disorder.c`
  - deterministic disorder application
- `rgf_payload.c`
  - JSON input parser
- `rgf_result.c`
  - JSON output writer
- `rgf_linalg.c`
  - BLAS/LAPACK wrappers
- `rgf_rng.c`
  - RNG utilities

### 6.2 Existing Python Files to Modify

- [cli.py](/home/msj/Desktop/playground/electroics/wtec/wtec/cli.py)
  - transport engine config
  - `wtec init` router prep
  - preflight checks
- [orchestrator.py](/home/msj/Desktop/playground/electroics/wtec/wtec/workflow/orchestrator.py)
  - dispatch to `kwant` or `rgf`
  - payload staging
  - PBS staging
  - result retrieval
- [cluster.py](/home/msj/Desktop/playground/electroics/wtec/wtec/config/cluster.py)
  - keep queue/source-of-truth logic reused
- [pbs.py](/home/msj/Desktop/playground/electroics/wtec/wtec/cluster/pbs.py)
  - keep PBS resource generation centralized
- [conductance.py](/home/msj/Desktop/playground/electroics/wtec/wtec/transport/conductance.py)
  - initially only as validation oracle and fallback
- [model.py](/home/msj/Desktop/playground/electroics/wtec/wtec/wannier/model.py)
  - reference logic only; do not tightly couple C code to Python internals

---

## 7. Submission Contract

The submission contract must be explicit because this package runs on PBS nodes,
not a generic local workstation.

### 7.1 Required PBS Shape

For any RGF run:

```bash
#PBS -N <job_name>
#PBS -q <queue>
#PBS -l select=1:ncpus=<queue_cores>:mpiprocs=<mpi_np>:ompthreads=1
#PBS -l walltime=<walltime>
#PBS -j oe
cd "$PBS_O_WORKDIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
mpirun -np <mpi_np> ./wtec_rgf_runner transport_payload.json transport_result.json
```

### 7.2 Not Allowed

- `python -m wtec ...` as the transport compute engine
- any `fork` launchstyle
- `mpi4py`
- hybrid thread/rank logic before it is validated

### 7.3 Rank Count Rule

The selected `mpi_np` is not:

- always 64 on `g4`
- always 48 on `g3`
- always 32 on `g2`
- always 16 on `g1`

It is:

`min(queue_cores, n_work_units, max_safe_ranks_from_memory)`

This rule must be implemented in:

- Python preflight
- runtime certificate output
- C startup validation

---

## 8. Build and Deployment Guidance

### 8.1 Build Location

The binary must be built on the cluster, not cross-compiled locally.

Reason:

- links against cluster MPI
- links against cluster BLAS/LAPACK/MKL
- avoids ABI mismatch

### 8.2 ISA Policy

Default build policy:

- conservative shared binary safe across all target queues

Optional later optimization:

- queue-family binaries if the queue CPUs are not ISA-identical

Do not default to:

- `-march=native`
- AVX-512-only binaries

### 8.3 `wtec init` Output

`wtec init` must leave the project in a state where a later `wtec run` can use
RGF without manual router repair.

That means `init` must:

- verify remote build prerequisites
- build or verify the binary
- stage the manifest
- store availability in `.wtec/init_state.json`

---

## 9. Validation Matrix

The validation matrix must be written before implementation starts, otherwise
the rewrite will drift into untestable claims.

### 9.1 Numerical Unit Tests

Required:

- 1D chain analytic transmission
- principal-layer invariance
- Sancho-Rubio convergence
- MPI gather correctness
- deterministic disorder reproducibility

### 9.2 Cross-Validation vs Kwant

Use current Kwant as the reference backend.

Reference cases:

1. nearest-neighbor toy chain
2. small multi-orbital SOC toy model
3. one real non-SOC HR
4. one real SOC HR

Compare:

- `G_mean`
- `G_std` when relevant
- runtime certificate sanity

### 9.3 Cluster Operational Validation

Required real-cluster proofs:

1. `wtec init` builds the binary on the cluster
2. `wtec run` submits through `qsub`
3. PBS launches through `mpirun`
4. result json is retrieved and parsed
5. the log shows:
   - queue
   - `mpi_np`
   - `P_eff`
   - `N_super`
   - work distribution

---

## 10. Performance Benchmark Plan

Performance claims should not be hardcoded into the code or the design.
They must be measured.

### 10.1 Phase-1 Benchmark Target

Benchmark only the mode that phase 1 truly supports:

- clean ordered `periodic_transverse`

Metrics:

- wall time
- peak RSS
- node utilization
- parity vs Kwant

Acceptance target:

- clear wall-time win over current Kwant on the same physically valid periodic
  case

### 10.2 Phase-2 Benchmark Target

Benchmark `full_finite` only after correctness and memory safety exist.

Metrics:

- fit/no-fit boundary by queue
- rank clipping behavior
- parity vs Kwant on small finite systems

---

## 11. Rollout Rules

The package must not jump from "design exists" to "Kwant deleted".

### Stage A

- `kwant` remains default
- `rgf` exists behind explicit config

### Stage B

- `auto` is added
- `auto` picks `rgf` only when:
  - mode is physically valid
  - binary exists
  - memory fit passes

### Stage C

- only after repeated cluster validation may clean periodic jobs default to RGF

### Stage D

- Kwant may be demoted only after both:
  - phase-1 parity
  - phase-2 finite viability

---

## 12. Failure Modes That Must Be Designed Up Front

The implementation must surface these explicitly.

### 12.1 Router Not Ready

Message should say:

- RGF binary unavailable
- run `wtec init` first

### 12.2 Geometry Not Valid for Periodic Reduction

Message should say:

- requested mode is `periodic_transverse`
- structure/disorder breaks periodicity on reduced axis
- choose `kwant` or explicit `full_finite`

### 12.3 Job Does Not Fit Queue Memory

Message should report:

- `n_orb`
- `P_eff`
- `N_super`
- per-rank memory
- requested ranks
- safe rank cap

### 12.4 Numerical Failure

Message should report:

- phase that failed:
  - payload parse
  - HR load
  - principal-layer sizing
  - Sancho-Rubio
  - sweep
  - gather

---

## 13. Minimal First Merge

The first merge request should be intentionally small.

It should contain only:

1. config/schema additions for transport engine selection
2. native project scaffold
3. `wtec init` router preparation hooks
4. no production dispatch yet

The first merge must not include:

- full solver
- auto cutover
- full finite mode
- deletion of Kwant

Reason:

- this keeps the review surface small
- it proves the router/build path independently from solver correctness

---

## 14. Recommended Development Order

Use this order exactly unless a blocking dependency forces a change.

1. schema freeze
2. native scaffold
3. `wtec init` router ownership
4. HR parser
5. `P_eff` computation
6. superslice Hamiltonian builder
7. Sancho-Rubio
8. RGF sweep
9. periodic reduction
10. Python integration
11. preflight memory clipping
12. numerical validation vs Kwant
13. cluster performance validation
14. experimental full finite mode
15. controlled default cutover

This order matters because it keeps:

- cluster operations testable early
- solver math testable before rollout
- full finite complexity out of the phase-1 critical path

---

## 15. Definition of Done

The rewrite is only "done" when all of the following are true:

1. `wtec init` prepares the RGF router automatically
2. `wtec run` can submit an RGF transport job through `qsub`
3. the job launches through `mpirun`
4. no fork launchstyle is involved anywhere
5. clean periodic transport matches Kwant numerically
6. runtime certificate records actual sizing and queue decisions
7. the implementation is element agnostic
8. full-node utilization is materially better for the periodic production path
9. Kwant remains available until the new path is proven

If any of these are missing, the rewrite is not complete.
