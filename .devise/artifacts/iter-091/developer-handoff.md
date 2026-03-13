# Iteration 91 Developer Handoff

## What Changed

- No new product code was retained this iteration.
- Current branch HEAD before this handoff commit: `c8b7e08c11ef916f2e9636e053ca1d53727c4a0d`
- The live benchmark continues on the thin-group fanout scheduler introduced by product commit `40eec77fe36760623f6b595ab50b7d889bd53c42`.

Rationale:

- I re-inspected the live continuity run and the current blocker is still the real benchmark computation, not a simple local-sync failure.
- The fresh evidence does not justify another unvalidated scheduler rewrite mid-run. The next safe move for the verifier is to keep polling the authoritative root while we preserve the evidence trail.

## Regression Evidence

Focused benchmark regressions on current HEAD:

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
7 passed in 0.62s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.23s
```

Declared dry-test contract rerun in this developer pass:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.39s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.23s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.21s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.26s
```

## Local Full-Finite Parity

Refreshed the checked-in toy parity workflow again on current HEAD. Result:

```json
{
  "abs_delta": 2.9999957497084395e-06,
  "has_runtime_cert": true,
  "kwant_t": 1.0,
  "mode": "full_finite",
  "progress_has_worker_done": true,
  "rgf_t": 0.9999970000042503
}
```

Interpretation:

- local toy parity still clears the `5e-6` acceptance bar
- `transport_results_raw.mode == "full_finite"` remains present
- a top-level `runtime_cert` remains present
- `progress.jsonl` still reaches `worker_done`

## Real Continuity Benchmark

Authoritative continuity root remains:

- `tmp/iter35_kwant_walltime`

Live local driver during this developer pass:

```text
PID 426185
.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Current real cluster jobs:

- Kwant reference: `60200`
- native RGF: `60201`

Current scheduler state is still the real chartered path:

- remote PBS `qsub`
- native `mpirun`
- no fork launchstyle for actual computation

### Kwant Reference Branch

The local authority checkpoint is still flat:

```json
{
  "results_len": 5,
  "status": "partial",
  "task_count_completed": 5,
  "task_count_expected": 35
}
```

Local shard visibility is still unchanged:

- `kwant_reference.rank0.jsonl`: present
- `kwant_reference.rank1.jsonl`: absent
- `kwant_reference.rank2.jsonl`: absent
- `kwant_reference.rank3.jsonl`: absent

Latest live local log evidence still shows the thin-group fanout scheduler and live heartbeats:

```text
[wtec][runtime] start 2026-03-13T16:03:48+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=27 load_max=10985 counts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 5, 5, 5] first_wave_thicknesses=[3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 9, 11, 13]
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=420.0
[kwant-bench][rank=9] heartbeat thickness_uc=5 energy_abs_ev=13.804600 elapsed_s=360.0
[kwant-bench][rank=10] heartbeat thickness_uc=7 energy_abs_ev=13.604600 elapsed_s=180.0
```

Fresh direct remote probe from this developer shell shows the current Kwant job is real but memory-heavy:

```text
Job 60200 -> exec_host = n019/64
n019.hpc
Mem: 377 total / 270 used / 106 free GiB
16 live python3 -m wtec.transport.kwant_nanowire_benchmark ranks
five ranks around 40.1 GiB RSS and ~290-298% CPU
five ranks around 14.6 GiB RSS and ~206-213% CPU
three ranks around 1.7-2.9 GiB RSS and ~98% CPU
three defunct helper processes also visible
```

Interpretation:

- the Kwant branch is not fake-idle; it is doing real cluster work
- the remaining question is throughput under substantial real memory pressure, not whether local log syncing is missing
- the authority state is still flat locally: no completion beyond `5/35`, no new shard files, and no fresh local `done` line yet

### Native-RGF Branch

The native side continues to advance and is not the dominant blocker.

Current local artifact state for `d07_e0p0`:

- `sigma_manifest.json`: present
- `transport_result.json`: absent
- `transport_runtime_cert.json`: absent

Current synced live `d07_e0p0` log:

```text
[wtec][runtime] start 2026-03-13T16:03:51+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T16:03:51+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=7 energy_ev=13.6046 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=8736 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
[wtec][sigma] selfenergy_left_done wall_seconds=409.443 solver=lopez_sancho iterations=22
[wtec][sigma] selfenergy_right_start lead_dim=8736 solver=lopez_sancho
```

Fresh direct remote probe for the native job:

```text
Job 60201 -> exec_host = n020/64
n020.hpc
Mem: 377 total / 18 used / 317 free GiB
python3 -m wtec.transport.kwant_sigma_extract ... thickness_uc=7 ... at ~5719% CPU with 127 threads and RSS ~17.6 GiB
```

Interpretation:

- the native-RGF side is still materially active on the real cluster path
- the native branch remains ahead of the Kwant branch in terms of continuity progress
- the native branch is not the acceptance blocker

## Still Missing

Benchmark-wide local authority outputs are still absent:

- `tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json`
- `tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json`
- `tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json`
- `tmp/iter35_kwant_walltime/benchmark_summary.json`

Kwant progress gaps still present locally:

- `kwant_reference.json` still only `5/35`
- `kwant_reference.rank1.jsonl` absent
- `kwant_reference.rank2.jsonl` absent
- `kwant_reference.rank3.jsonl` absent
- no fresh local `done` line observed in `model_b/c/kwant/wtec_job.log`

## What Remains Unresolved

- benchmark-wide parity against the completed Kwant reference path
- raw/fit tolerance compliance via `comparison_raw.json` and `comparison_fit.json`
- final `>=5x` wall-time speedup proof
- separate restart acceptance; the last verified blocker remains remote SIESTA preparation failing on missing ScaLAPACK

## What The Verifier Should Do Next

1. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`; do not reset the evidence trail.
2. Verify current HEAD before using this handoff.
3. Continue polling for the first fresh thin-group-fanout Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
4. Continue polling the native side until `d07_e0p0` writes:
   - `transport_result.json`
   - `transport_runtime_cert.json`
5. Do not clear acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
6. Treat the current scheduler question as a throughput-versus-memory tradeoff, not an observability bug:
   - the thin-group fanout run is real
   - the node is heavily occupied
   - the missing acceptance proof is still benchmark completion, not local sync
7. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
