# Iteration 85 Developer Handoff

## What Changed

- Product change: seed the live Kwant benchmark's first wave with lighter points before backfilling heavier work.
- Landed product commit: `bf3e53d` (`Seed kwant benchmark first wave with lighter tasks`).
- Files changed:
  - [`wtec/transport/kwant_nanowire_benchmark.py`](../../../../wtec/transport/kwant_nanowire_benchmark.py)
  - [`tests/test_kwant_nanowire_benchmark_resume.py`](../../../../tests/test_kwant_nanowire_benchmark_resume.py)

### Runtime change

The previous weighted scheduler still started several very heavy thickness-13 and thickness-11 points immediately, while even a thickness-3 point stayed alive for many minutes without producing the first fresh completion row. The new distributor now:

- seeds one initial task per rank from the globally lightest pending tasks
- then greedily backfills the remaining heavier tasks by current estimated load
- preserves deterministic ordering and local ascending-cost execution per rank
- logs the resulting first-wave thicknesses on rank 0 for live diagnosis

That keeps the continuity root on the same real runtime contract while lowering the first-wave memory/compute pressure and making early checkpoint movement more likely.

### New focused coverage

Added a new regression that proves the first wave is seeded from the globally lightest pending tasks, while retaining the prior balance and checkpoint-contract coverage.

## Focused Regressions

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
7 passed in 0.58s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.18s
```

## Declared Dry-Test Contract

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.33s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.25s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.18s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.25s
```

Interpretation:

- the new scheduler patch keeps the full declared regression surface green

## Local Full-Finite Parity

The checked-in toy full-finite parity workflow remains valid on this branch baseline:

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

- toy parity still clears the `5e-6` bar
- `transport_results_raw.mode == "full_finite"` remains present
- a top-level `runtime_cert` remains present
- `progress.jsonl` still reaches `worker_done`

## Real Continuity Root

Authoritative continuity root:

- `tmp/iter35_kwant_walltime`

Fresh local driver after the lighter-first relaunch:

```text
PID 397617
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh live jobs:

- Kwant: `60185` on `n019`
- native RGF: `60186` on `n021`

The relaunch preserved the continuity root correctly:

- partial Kwant checkpoint was reused
- cached native results through `d07_em0p1` were reused
- the fresh native frontier is now `d07_e0p0`

## New Kwant Runtime Evidence

The new lighter-first scheduler is now active on the real PBS `qsub` + native `mpirun` path with no fork launchstyle.

Fresh local log evidence:

```text
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=343 load_max=2224 counts=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1] first_wave_thicknesses=[3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9]
[kwant-bench][rank=0] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=1] start thickness_uc=3 energy_abs_ev=13.504600
[kwant-bench][rank=2] start thickness_uc=3 energy_abs_ev=13.704600
[kwant-bench][rank=3] start thickness_uc=3 energy_abs_ev=13.404600
[kwant-bench][rank=4] start thickness_uc=3 energy_abs_ev=13.804600
[kwant-bench][rank=5] start thickness_uc=5 energy_abs_ev=13.604600
[kwant-bench][rank=10] start thickness_uc=7 energy_abs_ev=13.604600
[kwant-bench][rank=15] start thickness_uc=9 energy_abs_ev=13.604600
```

Latest local heartbeat evidence from the same fresh run:

```text
[kwant-bench][rank=4] heartbeat thickness_uc=3 energy_abs_ev=13.804600 elapsed_s=140.0
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=140.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=140.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=140.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=140.0
[kwant-bench][rank=5] heartbeat thickness_uc=5 energy_abs_ev=13.604600 elapsed_s=40.0
[kwant-bench][rank=8] heartbeat thickness_uc=5 energy_abs_ev=13.404600 elapsed_s=40.0
[kwant-bench][rank=10] start thickness_uc=7 energy_abs_ev=13.604600
```

Interpretation:

- this relaunch materially changes the first-wave runtime shape versus the prior heavier-first patch
- the first live wave is now all thickness `3/5/7/9`, rather than immediately launching thickness `13/11` points
- the local authority checkpoint is still flat at `5/35`, and no fresh local `done` line or new shard file landed during this short observation window
- the new scheduler is therefore active, but the benchmark-wide acceptance blocker is still unresolved

## Native-RGF Continuity Evidence

The native side is still not the dominant blocker.

What the relaunch reused:

- full thickness-1 slice
- full thickness-3 slice
- full thickness-5 slice
- `d07_em0p2`
- `d07_em0p1`

Current fresh native frontier:

- `d07_e0p0`

Current local artifact state for `d07_e0p0`:

- `transport_result.json`: absent
- `transport_runtime_cert.json`: absent
- `sigma_manifest.json`: absent

Current live `d07_e0p0` log:

```text
[wtec][runtime] start 2026-03-13T15:32:13+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T15:32:13+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=7 energy_ev=13.6046 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=8736 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
```

Interpretation:

- actual heavy native-RGF transport computation continues on the real cluster path
- the live frontier has moved beyond `d07_em0p1` into `d07_e0p0`
- native continuity is still advancing and remains secondary to the Kwant blocker

## Still Missing

Benchmark-wide local authority outputs are still absent:

- `tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json`
- `tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json`
- `tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json`
- `tmp/iter35_kwant_walltime/benchmark_summary.json`

Kwant progress gaps still present locally:

- `kwant_reference.json` still reports `5/35`
- only `kwant_reference.rank0.jsonl` exists locally
- no fresh local `done` line has landed yet in `model_b/c/kwant/wtec_job.log`

Meaning:

- benchmark-wide parity is still unproven
- raw/fit tolerance compliance is still unproven
- the required `>=5x` speedup remains unproven

## Restart Status

I did **not** rerun `wtec init` after this product patch.

Last verified restart status remains unchanged from the prior debugger rerun:

- optional warning:
  - `meson-python: error: meson executable "meson" not found`
- material blocker:
  - `Could NOT find CustomScalapack (missing: SCALAPACK_LIBRARY)`
  - `MPI was requested, but ScaLAPACK could not be found by CMake`

Interpretation:

- restart acceptance remains independently blocked by the same remote SIESTA / ScaLAPACK defect

## What The Verifier Should Do Next

1. Verify the lighter-first scheduler product commit and the current HEAD that records this handoff.
2. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`; do not reset the evidence trail.
3. Confirm the fresh local Kwant log now shows:
   - `first_wave_thicknesses=[3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9]`
   - first live ranks starting on thickness `3/5/7/9` rather than `13/11`
4. Continue polling for the first fresh Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
5. Continue polling the native side until `d07_e0p0` returns its local artifact set.
6. Do not claim acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
7. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
