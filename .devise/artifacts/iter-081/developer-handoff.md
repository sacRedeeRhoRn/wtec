# Iteration 81 Developer Handoff

## What Changed

- Product commit: `a5b3762` (`Balance kwant benchmark work across ranks`)
- Scope:
  - replaced the Kwant benchmark's static `pending_tasks[rank::size]` split with a deterministic weighted scheduler in [`wtec/transport/kwant_nanowire_benchmark.py`](../../../../wtec/transport/kwant_nanowire_benchmark.py)
  - the new scheduler uses a thickness-based cost proxy, spreads heavier points across ranks first, and orders each rank's local queue from lighter to heavier points so partial checkpoint rows can land earlier without changing the checkpoint contract
  - added a focused regression in [`tests/test_kwant_nanowire_benchmark_resume.py`](../../../../tests/test_kwant_nanowire_benchmark_resume.py) proving the new scheduler preserves the task set, reduces the worst-rank estimated load relative to the old strided split, and keeps each local queue ordered by increasing estimated cost
- No unrelated repo changes were touched.

## Why This Change

- The live real-cluster `16x4` Kwant run on the authoritative continuity root was still flat locally at `5/35` even after the round-barrier removal.
- Remote evidence showed the current run was genuinely computing, not idling:
  - remote workdir still contained only `kwant_reference.rank0.jsonl`, so the blocker was not a local retrieval miss
  - the real `16x4` run had all 16 ranks alive, but the old strided task split front-loaded very uneven thickness bands per rank
- For the current pending sweep (`thicknesses = 3,5,7,9,11,13` with five energies each), the old split effectively assigned contiguous thickness groups to ranks, which creates avoidable stragglers on the thickest points.
- The new scheduler is a durable runtime improvement: it does not relax correctness, it does not change physics, and it directly targets the benchmark wall-time objective by balancing the expensive Kwant points across the live MPI ranks.

## Focused Regressions

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
6 passed in 1.04s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.56s
```

## Declared Dry-Test Contract

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.88s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.43s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.36s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.33s
```

Interpretation:

- the full declared regression surface is green on the product commit
- the new Kwant scheduler is covered by focused resume/runtime tests

## Local Full-Finite Parity

The checked-in toy full-finite workflow still clears the required local parity bar:

```json
{
  "rgf_t": 0.9999970000042503,
  "kwant_t": 1.0,
  "abs_delta": 2.9999957497084395e-06,
  "mode": "full_finite",
  "has_runtime_cert": true,
  "progress_has_worker_done": true
}
```

Interpretation:

- local toy parity still clears the `5e-6` acceptance bar
- `transport_results_raw.mode == "full_finite"` remains present
- a top-level `runtime_cert` remains present
- `progress.jsonl` still reaches `worker_done`

## Real Continuity Root

Authoritative continuity root:

- `tmp/iter35_kwant_walltime`

Fresh relaunched local driver:

```text
PID 376065
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh live jobs:

- Kwant: `60175` on `n019`
- native RGF: `60176` on `n021`

The relaunch preserved continuity correctly:

- it resumed the partial Kwant reference already stored under `model_b/c/kwant/kwant_reference.json`
- it reused all cached native-RGF points through the full thickness-1, thickness-3, and thickness-5 slices
- it also reused the fresh thickness-7 `d07_em0p2` artifact set after that job finished and synced locally

## New Kwant Runtime Evidence

The new weighted scheduler is active on the real PBS `qsub` + native `mpirun` path with no fork launchstyle.

Fresh local Kwant log excerpt after the relaunch:

```text
[wtec][runtime] start 2026-03-13T15:09:18+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=979 load_max=2197 counts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 7, 3]
[kwant-bench][rank=0] start thickness_uc=13 energy_abs_ev=13.604600
[kwant-bench][rank=1] start thickness_uc=13 energy_abs_ev=13.504600
[kwant-bench][rank=2] start thickness_uc=13 energy_abs_ev=13.704600
[kwant-bench][rank=3] start thickness_uc=13 energy_abs_ev=13.404600
[kwant-bench][rank=5] start thickness_uc=11 energy_abs_ev=13.604600
[kwant-bench][rank=10] start thickness_uc=7 energy_abs_ev=13.404600
[kwant-bench][rank=12] start thickness_uc=5 energy_abs_ev=13.604600
[kwant-bench][rank=14] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=14] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=20.0
[kwant-bench][rank=14] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=40.0
```

Interpretation:

- the new scheduler materially changed the first-wave runtime shape on the real cluster path
- unlike the old strided split, the fresh start wave now mixes `thickness_uc = 13, 11, 7, 5, 3` across the 16 ranks instead of grouping the early wave by contiguous thickness bands
- the current local checkpoint is still partial at `5/35`, and only `kwant_reference.rank0.jsonl` is visible locally so far
- no fresh `done` row landed during this short observation window, so the benchmark-wide blockage is not yet cleared

## Native-RGF Continuity Evidence

The native side is not the dominant blocker.

What changed during this pass:

- the previous live `d07_em0p2` job finished cleanly and its local artifact set is now present:
  - `transport_result.json`
  - `transport_runtime_cert.json`
  - `sigma_manifest.json`
- representative runtime-cert evidence for `d07_em0p2`:

```json
{
  "wall_seconds": 123.842437,
  "effective_thread_count": 51.791766,
  "queue": "g4",
  "mpi_size": 1,
  "omp_threads": 64
}
```

- the fresh native frontier after the relaunch is now `d07_em0p1`

Current live `d07_em0p1` log:

```text
[wtec][runtime] start 2026-03-13T15:09:21+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T15:09:21+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=7 energy_ev=13.5046 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=8736 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
```

Interpretation:

- the native-RGF side continues to run on the real cluster path and keeps advancing
- the real blocker remains the incomplete Kwant reference / benchmark aggregation side

## Still Missing

Benchmark-wide local authority outputs are still absent:

- `tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json`
- `tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json`
- `tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json`
- `tmp/iter35_kwant_walltime/benchmark_summary.json`

Kwant progress gaps still present locally after the relaunch:

- `kwant_reference.json` still reports `5/35`
- only `kwant_reference.rank0.jsonl` exists locally
- no fresh `done` line has landed yet in `model_b/c/kwant/wtec_job.log`

## Restart Status

I did **not** rerun `wtec init` after the product patch in this developer pass.

Last verified restart status remains unchanged from the prior debugger rerun:

- optional warning while preparing the solver backend:
  - `meson-python: error: meson executable "meson" not found`
- material restart blocker:
  - `Could NOT find CustomScalapack (missing: SCALAPACK_LIBRARY)`
  - `MPI was requested, but ScaLAPACK could not be found by CMake`

Interpretation:

- restart acceptance remains independently blocked by the same remote SIESTA / ScaLAPACK defect

## What The Verifier Should Do Next

1. Verify commit `a5b3762` and the current HEAD that records this handoff.
2. Recheck the authoritative continuity root `tmp/iter35_kwant_walltime` rather than starting a fresh benchmark root.
3. Confirm the new Kwant distribution summary is present in the live local `wtec_job.log` and that the relaunch really starts mixed thicknesses across ranks.
4. Keep polling for the first fresh Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
5. Confirm the native side keeps reusing cached slices and that `d07_em0p1` eventually returns its local artifact set.
6. Do not claim acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
7. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
