# Iteration 9 Developer Handoff

## Verdict

- Status: `needs_fix`
- Branch: `devise/wtec-rgf-completion/developer`
- Declared dry tests: green
- Real benchmark relaunched on this commit: yes
- Goal fully met: no

This iteration changes the benchmark orchestrator so the primary-model native-RGF benchmark path is no longer hard-gated behind full Kwant completion in the code path. The declared full-finite dry-test contract remains green, and focused coverage now asserts the new overlap helper ordering. In live use, the patched command did reach the new overlap branch twice, but both relaunches still idled locally before any native-RGF run directory or transport artifact appeared. I cancelled those runs after capturing evidence so they would not keep burning cluster time.

## What Changed

### 1. Factored the primary-model RGF axis loop into a helper

File:

- `wtec/cli.py`

New helper:

- `_run_rgf_benchmark_axis(...)`

Purpose:

- centralizes the existing per-thickness / per-energy native-RGF benchmark loop
- preserves the existing transport contract:
  - `transport_backend = "qsub"`
  - `transport_engine = "rgf"`
  - `transport_rgf_mode = "full_finite"`
  - `transport_rgf_full_finite_sigma_backend = "native"`
  - `transport_rgf_full_finite_kwant_script = ""`
- keeps the per-point `transport_result.json` / `transport_runtime_cert.json` generation path unchanged

### 2. Added a narrow overlap helper for Kwant + RGF benchmark orchestration

File:

- `wtec/cli.py`

New helper:

- `_run_kwant_and_rgf_overlap(...)`

Behavior:

- submits the blocking Kwant reference path in a one-worker `ThreadPoolExecutor`
- runs the primary-model native-RGF benchmark axis loop on the main thread
- joins the Kwant future only after the RGF loop returns

Intent:

- once source artifacts, canonical HR/WIN, and `length_uc` are known, Kwant and native RGF are independent enough to advance in parallel
- this removes the previous CLI-level sequencing that forced the benchmark to wait for the entire Kwant baseline before starting any native-RGF transport work

### 3. Tightened the overlap branch to suppress background Kwant live-log streaming

File:

- `wtec/cli.py`

Change:

- in the overlap branch, the background `submit_kwant_nanowire_reference(...)` call now runs with `live_log=False`

Why:

- the first live relaunch showed that letting the background Kwant submit path stream logs while the main thread was supposed to enter native RGF was not a safe operational assumption for this managed driver context
- the quiet background branch keeps the overlap code path narrower and avoids dual live-log streams competing inside the same local benchmark driver

### 4. Added focused unit coverage for the overlap ordering contract

File:

- `tests/test_nanowire_benchmark.py`

New test:

- `test_run_kwant_and_rgf_overlap_runs_rgf_while_kwant_waits`

Coverage intent:

- proves the helper starts the Kwant submit path
- proves the RGF callback is executed before the Kwant future is joined
- locks the intended orchestration order without needing a live cluster in test

## Test Evidence

Focused benchmark tests:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "run_kwant_and_rgf_overlap or select_benchmark_models_defaults_to_primary_rgf_model or build_tis_benchmark_source_cfg_uses_explicit_source_nodes or tis_"
.....                                                                    [100%]
5 passed, 6 deselected in 0.28s
```

Declared dry-test contract:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
..                                                                       [100%]
2 passed, 3 deselected in 2.65s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
..                                                                       [100%]
2 passed, 8 deselected in 0.16s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
.....                                                                    [100%]
5 passed, 36 deselected in 0.32s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
......                                                                   [100%]
6 passed in 0.17s
```

## Real Benchmark Evidence

### Before this commit

The active pre-patch multi-rank Kwant run was still stalled in the same early state:

- local PID: `4127175`
- remote job: `59951`
- anchored active log segment: `2026-03-13T06:55:47+09:00`
- observed state before cancellation:
  - `started = 9`
  - `done = 5`
  - no `kwant_reference.json`
  - no `transport_result.json`
  - no `transport_runtime_cert.json`
  - no `benchmark_summary.json`

I preserved the old workspace before replacing it:

- `tmp/devise_transport_benchmark/model_b/c/kwant_overlap_snapshot_20260313T071341`

### On this commit

I relaunched the real benchmark twice from the existing source artifacts:

1. First overlap relaunch:
   - local PID: `4144031`
   - remote job: `59957`
   - local benchmark output reached:

   ```text
   [benchmark] model=model_b axis=c: launching Kwant and native RGF in parallel
   ```

   Observed blocker:

   - no `tmp/devise_transport_benchmark/model_b/c/rgf/...` directory ever appeared
   - process held exactly one live SSH socket to `202.30.0.129:54329`
   - local process threads were mostly sleeping on `futex_wait_queue`
   - the only remote live benchmark job was still the Kwant job

   I preserved that workspace too:

   - `tmp/devise_transport_benchmark/model_b/c/kwant_overlap_quiet_snapshot_20260313T071900`

2. Second overlap relaunch after silencing background Kwant live logs:
   - local PID: `4148705`
   - remote job: `59961`
   - local benchmark output again reached:

   ```text
   [benchmark] model=model_b axis=c: launching Kwant and native RGF in parallel
   ```

   Observed blocker remained the same:

   - still no `tmp/devise_transport_benchmark/model_b/c/rgf/...` path
   - still no benchmark transport artifacts
   - queue still showed only the Kwant job for this benchmark branch
   - the local driver again settled into a sleeping wait state without producing native-RGF benchmark output

I cancelled both relaunched jobs after capturing the state:

- `59957` -> completed/cancelled
- `59961` -> completed/cancelled

## What This Iteration Cleared

- The primary-model benchmark code path is no longer sequential by construction; the CLI now has an explicit seam for overlapping Kwant submission and native-RGF work.
- The overlap orchestration order is unit-tested.
- The declared full-finite regression gates remain green on this branch.

## What Remains Unresolved

- The live overlap relaunches still did not reach the first native-RGF benchmark run directory in the real workspace.
- No benchmark `kwant_reference.json` exists yet.
- No benchmark native-RGF `transport_result.json` exists yet.
- No benchmark native-RGF `transport_runtime_cert.json` exists yet.
- No `benchmark_summary.json` exists yet.
- The `>= 5x` benchmark speedup claim remains unproven.
- `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python` is still non-green because remote SIESTA preparation still fails on missing ScaLAPACK.

## What The Debugger Should Do Next

1. Inspect why the live overlap branch reaches the CLI message

   ```text
   [benchmark] model=model_b axis=c: launching Kwant and native RGF in parallel
   ```

   but never creates any `model_b/c/rgf/...` run directory before idling.

2. Treat the likely fault boundary as the local overlap driver rather than the remote Kwant job itself, because:

   - the Kwant job still submits successfully
   - the local process never reaches the first RGF transport directory creation point
   - only the Kwant side shows up remotely

3. Reproduce under a debugger or stack dump if possible on the live branch `a7e9c41..HEAD`, focusing on:

   - `_run_kwant_and_rgf_overlap(...)` in `wtec/cli.py`
   - the call from that helper into `_run_rgf_benchmark_axis(...)`
   - whether the main thread is parking before or during the first `_stage_transport_rgf_qsub(...)` invocation

4. Once that overlap-driver blocker is fixed, rerun the benchmark from the existing source artifacts and verify:

   - the first `model_b/c/rgf/.../transport/primary` directory appears locally
   - the first native-RGF benchmark `transport_result.json` / `transport_runtime_cert.json` appears
   - the Kwant and RGF paths both stay on real PBS `qsub` + native `mpirun`

5. Keep the snapshot directories above; they preserve the successive pre-fix benchmark states without overwriting the live `kwant` workspace path.
