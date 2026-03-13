# Iteration 79 Developer Handoff

## Scope

This pass focused on the lagging Kwant reference branch inside the authoritative continuity benchmark root `tmp/iter35_kwant_walltime`.

The concrete defect found in the benchmark code was not more log plumbing. It was the worker loop itself:

- each MPI rank owned a static slice of pending tasks
- but every rank still stopped at a per-round `comm.gather(...)` barrier
- faster ranks could not start their next task until the slowest rank in the current wave finished
- with thickness-dependent Kwant cost, that synchronization point can flatten both throughput and observable checkpoint progress

## Product Change

Committed product fix:

- `4cedda76f7d66ecaa2f065a75701909371fd95c2` `Remove kwant benchmark round barriers`

What changed:

- added `_run_local_tasks(...)` in `wtec.transport.kwant_nanowire_benchmark`
- each rank now runs its local pending-task slice straight through, appending shard rows as each task completes
- removed the old per-round MPI `gather` barrier that previously forced every rank to wait for the slowest task in the wave before any rank could advance to its next assigned task
- kept the existing shard/checkpoint contract:
  - per-rank shard rows still write immediately on task completion
  - rank 0 still updates the main checkpoint as its own rows complete
  - final merged checkpoint and validation still happen on rank 0
- added a focused regression covering the new local-task helper and shard append behavior

Files changed:

- `wtec/transport/kwant_nanowire_benchmark.py`
- `tests/test_kwant_nanowire_benchmark_resume.py`

## Focused Regressions

Focused benchmark regressions on the new worker-loop path:

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
5 passed in 0.59s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.22s
```

## Declared Dry-Test Contract

All configured dry-test commands passed again after the patch:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 5.99s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.38s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.29s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.37s
```

## Local Full-Finite Parity

Rebuilt the native runner and refreshed the checked-in toy parity workflow.

Note:

- the payload still uses a relative HR path, so the runner must be invoked from `tmp/devise_rgf_full_finite/`

Final result:

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
- `transport_results_raw.mode == "full_finite"` is still present
- a top-level `runtime_cert` is still present
- `progress.jsonl` still reaches `worker_done`

## Real Continuity Benchmark

### Authoritative run

I retired the old pre-patch continuity attempt and relaunched the same authoritative root on the patched worker loop:

- root: `tmp/iter35_kwant_walltime`
- live local driver PID: `358094`
- command:

```text
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Current live remote jobs:

- Kwant: `60163`, queue `g4`, host `n019`, walltime `03:00:00`
- native RGF: `60164`, queue `g4`, host `n020`, walltime `01:00:00`

This run is still on the chartered real path:

- remote PBS `qsub`
- native `mpirun`
- no fork launchstyle for actual transport computation

### Native side

The relaunch reused all cached native results through:

- the full thickness-1 slice
- the full thickness-3 slice
- the full thickness-5 slice (`d05_em0p2`, `d05_em0p1`, `d05_e0p0`, `d05_e0p1`, `d05_e0p2`)

The fresh native frontier is now:

- `d07_em0p2`

Current local artifact state for that fresh point:

```text
tmp/iter35_kwant_walltime/model_b/c/rgf/d07_em0p2/transport/primary/transport_result.json         missing
tmp/iter35_kwant_walltime/model_b/c/rgf/d07_em0p2/transport/primary/transport_runtime_cert.json   missing
tmp/iter35_kwant_walltime/model_b/c/rgf/d07_em0p2/transport/primary/sigma_manifest.json            missing
```

Current local runtime log for the fresh attempt:

```text
[wtec][runtime] start 2026-03-13T14:48:00+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T14:48:00+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=7 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=8736 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
```

### Kwant side

The patched Kwant branch relaunched from the same partial reference checkpoint:

- still `status == "partial"`
- still `task_count_completed == 5`
- still `task_count_expected == 35`

Local shard visibility at the end of this pass:

- `kwant_reference.rank0.jsonl`: present
- `kwant_reference.rank1.jsonl`: absent
- `kwant_reference.rank2.jsonl`: absent
- `kwant_reference.rank3.jsonl`: absent

Fresh real-runtime log evidence from the patched run:

```text
[wtec][runtime] start 2026-03-13T14:47:57+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench][rank=0] start thickness_uc=3 energy_abs_ev=13.404600
[kwant-bench][rank=1] start thickness_uc=3 energy_abs_ev=13.504600
[kwant-bench][rank=2] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=3] start thickness_uc=3 energy_abs_ev=13.704600
[kwant-bench][rank=4] start thickness_uc=3 energy_abs_ev=13.804600
[kwant-bench][rank=5] start thickness_uc=5 energy_abs_ev=13.404600
[kwant-bench][rank=6] start thickness_uc=5 energy_abs_ev=13.504600
[kwant-bench][rank=7] start thickness_uc=5 energy_abs_ev=13.604600
[kwant-bench][rank=8] start thickness_uc=5 energy_abs_ev=13.704600
[kwant-bench][rank=9] start thickness_uc=5 energy_abs_ev=13.804600
[kwant-bench][rank=10] start thickness_uc=7 energy_abs_ev=13.404600
[kwant-bench][rank=11] start thickness_uc=7 energy_abs_ev=13.504600
[kwant-bench][rank=12] start thickness_uc=7 energy_abs_ev=13.604600
[kwant-bench][rank=13] start thickness_uc=7 energy_abs_ev=13.704600
[kwant-bench][rank=14] start thickness_uc=7 energy_abs_ev=13.804600
[kwant-bench][rank=15] start thickness_uc=9 energy_abs_ev=13.404600
```

and the job is still heartbeating per rank during those long solves:

```text
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=400.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=400.0
[kwant-bench][rank=5] heartbeat thickness_uc=5 energy_abs_ev=13.404600 elapsed_s=320.2
[kwant-bench][rank=8] heartbeat thickness_uc=5 energy_abs_ev=13.704600 elapsed_s=320.0
[kwant-bench][rank=10] heartbeat thickness_uc=7 energy_abs_ev=13.404600 elapsed_s=141.5
[kwant-bench][rank=14] heartbeat thickness_uc=7 energy_abs_ev=13.804600 elapsed_s=140.0
```

Interpretation:

- the patched worker loop is live on the real cluster path
- the benchmark still has not produced the first post-patch completed Kwant task during this observation window
- therefore the absence of new shard files is still explained by task runtime, not by missing live-sync plumbing
- the code change is meant to matter when the first ranks begin finishing: they can now move directly to their next assigned task instead of waiting for the slowest rank in the wave

## Still Missing

These benchmark-wide local authority outputs remain absent:

```text
tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json
tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json
tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json
tmp/iter35_kwant_walltime/benchmark_summary.json
```

So benchmark-wide parity, raw/fit tolerance compliance, and the required `>=5x` speedup proof remain unresolved.

## Restart

I did not rerun `wtec init` after the worker-loop patch in this developer pass.

The last verified restart blocker remains unchanged from the prior debugger pass:

- optional warning: `meson-python: error: meson executable "meson" not found`
- material failure: remote SIESTA preparation still cannot find ScaLAPACK

## Verifier Next

1. Continue from the authoritative continuity root `tmp/iter35_kwant_walltime` and the live local driver PID `358094`.
2. Poll for the first fresh post-patch Kwant completion:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - `done` lines in `model_b/c/kwant/wtec_job.log`
3. Once a first fresh Kwant completion lands, verify the runtime consequence of the patch:
   - a faster rank should be able to emit the next `start ...` line for its second assigned task without waiting for the slowest rank in the original wave
4. Continue polling the native side until `d07_em0p2` returns its final local artifact set.
5. Do not claim success until the benchmark-wide authority files exist locally:
   - `model_b/c/kwant/kwant_reference.json` beyond the current `5/35`
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
6. Keep the SIESTA / ScaLAPACK restart defect separate; it remains an independent acceptance blocker.
