# Iteration 7 Developer Handoff

## Verdict

- Status: `needs_verify`
- Branch: `devise/wtec-rgf-completion/developer`
- Benchmark job relaunched: yes
- Declared dry tests: green
- Goal fully met: not yet

This iteration keeps the declared full-finite regression gates green and changes the real Kwant benchmark stage from a single-rank `64x1` layout to a conservative multi-rank `16x4` layout that matches the task-parallel structure of the Kwant payload. The benchmark is live again on the real PBS `qsub` + native `mpirun` path, and the new run is already executing four Kwant points concurrently. The acceptance bar is still open because the live run has not yet finished the Kwant baseline, has not yet entered the native-RGF transport stage, and therefore still has no `kwant_reference.json`, `transport_result.json`, `transport_runtime_cert.json`, or `benchmark_summary.json`.

## What Changed

### 1. Parallelized the real Kwant reference stage conservatively

File:

- `wtec/transport/nanowire_benchmark_cluster.py`

Changes:

- Added `_kwant_worker_layout(...)` to choose a bounded MPI/OMP layout for the independent Kwant benchmark tasks.
- Default behavior now uses up to `4` MPI ranks with the remaining node cores split evenly across OMP threads.
- Added optional env override `TOPOSLAB_KWANT_BENCH_MPI_RANKS` for controlled experiments.
- Updated `submit_kwant_nanowire_reference(...)` so the generated PBS script now requests:
  - `mpiprocs = mpi_np / n_nodes`
  - `ompthreads = total_cores / mpi_np`
  - `mpirun -np <mpi_np> --bind-to none`
- Preserved the required runtime path:
  - real PBS `qsub`
  - real native `mpirun`
  - no fork launchstyle

Rationale:

- The Kwant payload already partitions independent `(thickness, energy)` points across MPI ranks in `wtec.transport.kwant_nanowire_benchmark`.
- The previous live benchmark was running a full `64` cores as `1 MPI x 64 OMP`, which advanced too slowly to reach the native-RGF stage in a practical turnaround.
- The new `4 MPI x 16 OMP` layout keeps MUMPS pressure bounded while exploiting the task-level parallelism already present in the payload.

### 2. Locked the PBS-layout contract with regression coverage

File:

- `tests/test_nanowire_benchmark_cluster.py`

Changes:

- Updated the cluster-script regression to assert the default one-node Kwant launch now uses:
  - `#PBS -l select=1:ncpus=64:mpiprocs=4:ompthreads=16`
  - `mpirun -np 4 --bind-to none`
  - `OMP/MKL/OPENBLAS/NUMEXPR = 16`
- Added coverage for `TOPOSLAB_KWANT_BENCH_MPI_RANKS=2`, asserting a `2 MPI x 32 OMP` layout.

## Dry-Test Evidence

Fresh results on this iteration:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
..                                                                       [100%]
2 passed, 3 deselected in 2.48s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
..                                                                       [100%]
2 passed, 8 deselected in 0.11s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
.....                                                                    [100%]
5 passed, 36 deselected in 0.12s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
......                                                                   [100%]
6 passed in 0.14s
```

Focused supporting checks during implementation:

```text
pytest -q tests/test_nanowire_benchmark_cluster.py tests/test_nanowire_benchmark_progress.py
6 passed in 0.17s

pytest -q tests/test_pbs_resources.py tests/test_nanowire_benchmark.py -k 'tis_ or source_nodes or select_benchmark_models'
4 passed, 13 deselected in 0.12s
```

## Real Benchmark Evidence

### Relaunch details

- Old slow Kwant job:
  - remote job `59946`
  - single-rank log signature: `mpi=1 threads=64`
- Old local benchmark process was terminated after preserving the workspace snapshot.
- Prior single-rank workspace snapshot preserved at:
  - `tmp/devise_transport_benchmark/model_b/c/kwant_single_rank_snapshot_20260313T065522`

### Current live run

- Current local benchmark PID: `4127175`
- Current remote Kwant job: `59951`
- Command:

```text
.venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/devise_transport_benchmark --queue g4 --walltime 01:00:00
```

- The relaunched benchmark reused the already-generated source artifacts:
  - `tmp/devise_transport_benchmark/model_b/source_artifacts.json`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS_hr.dat`

### Current remote PBS script evidence

Remote script:

- `/home/msj/Desktop/playground/electroics/wtec/remote_runs/nanowire_benchmark/mp-1018028/model_b_c_kwant/kwant_reference.pbs`

Observed key lines:

```text
#PBS -l select=1:ncpus=64:mpiprocs=4:ompthreads=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
mpirun -np 4 --bind-to none env PYTHONPATH=$PWD/wtec_src.zip:$PYTHONPATH python3 -m wtec.transport.kwant_nanowire_benchmark kwant_payload.json kwant_reference.json
```

This is direct evidence that the real benchmark remains on the required runtime path:

- PBS `qsub`
- native `mpirun`
- no fork launchstyle

### Current live log evidence

Newest run banner and concurrency from the active `wtec_job.log`:

```text
[wtec][runtime] start 2026-03-13T06:55:47+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=4 threads=16 length_uc=24
[kwant-bench][rank=0] start thickness_uc=1 energy_abs_ev=13.404600
[kwant-bench][rank=1] start thickness_uc=1 energy_abs_ev=13.504600
[kwant-bench][rank=2] start thickness_uc=1 energy_abs_ev=13.604600
[kwant-bench][rank=3] start thickness_uc=1 energy_abs_ev=13.704600
[kwant-bench][rank=3] done thickness_uc=1 energy_abs_ev=13.704600 transmission=43.999999999999
[kwant-bench][rank=2] done thickness_uc=1 energy_abs_ev=13.604600 transmission=40.000000000000
[kwant-bench][rank=0] done thickness_uc=1 energy_abs_ev=13.404600 transmission=34.000000000001
[kwant-bench][rank=1] done thickness_uc=1 energy_abs_ev=13.504600 transmission=37.999999999997
```

Interpretation:

- The relaunch is materially healthier than the prior `mpi=1 threads=64` run.
- Four independent Kwant points are now executing concurrently on the real cluster.
- The benchmark is still in the Kwant baseline stage, not yet in the native-RGF transport stage.

## Still Unresolved

- No local `tmp/devise_transport_benchmark/**/kwant_reference.json` yet.
- No local benchmark `transport_result.json` yet.
- No local benchmark `transport_runtime_cert.json` yet.
- No local `benchmark_summary.json` yet.
- No measured benchmark parity or `>= 5x` speedup proof yet.
- The configured restart command is still operationally incomplete because `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python` still fails in remote SIESTA preparation when ScaLAPACK is missing.

## What The Debugger Should Do Next

1. Let remote job `59951` complete and verify that it writes `kwant_reference.json`.
2. Confirm the benchmark then transitions into the native-RGF transport stage on the same real PBS `qsub` + native `mpirun` path.
3. Wait for the first complete benchmark artifact set:
   - `kwant_reference.json`
   - `transport_result.json`
   - `transport_runtime_cert.json`
   - `benchmark_summary.json`
4. Use `benchmark_summary.json` to measure the actual Kwant-vs-RGF wall-time ratio before claiming the required `>= 5x` speedup.
5. Treat the active remote `wtec_job.log` as append-only across retries; the file still contains cancelled-job lines from `59946`, so verification should anchor on the later `2026-03-13T06:55:47+09:00` restart banner and the `mpi=4 threads=16` signature.
