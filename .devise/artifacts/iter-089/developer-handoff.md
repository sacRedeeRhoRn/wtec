# Iteration 89 Developer Handoff

## What Changed

- Product commit: `40eec77` (`Split kwant benchmark thin groups across ranks`)
- Current handoff commit target before this file is committed: `40eec77`

Changed files:

- `wtec/transport/kwant_nanowire_benchmark.py`
- `tests/test_kwant_nanowire_benchmark_resume.py`

Behavior change:

- kept the grouped-thickness memory-saving contract for the Kwant benchmark
- changed the `world_size >= len(grouped_tasks)` branch so extra ranks are spent on the cheapest thickness groups first instead of leaving many ranks idle
- retained thickness-local ordering inside each split bucket
- for the canonical resumed benchmark set (`30` pending tasks across thicknesses `3/5/7/9/11/13`, `mpi=16`) the first live wave is now:

```text
[3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 9, 11, 13]
```

Why this change:

- the prior grouped-thickness scheduler removed duplicated heavy systems, but it only used one rank per thickness group and left the widened `16x4` layout mostly idle
- this patch preserves the thickness-local reuse benefit while restoring additional task-level parallelism on the lightest groups first

## Regression Evidence

Focused benchmark regressions passed on the patched code:

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
7 passed in 0.52s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.18s
```

Declared dry-test contract also passed:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.07s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.21s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.18s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.20s
```

## Local Full-Finite Parity

Rebuilt the native runner and refreshed the checked-in toy parity workflow. Result:

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
- `transport_results_raw.mode == "full_finite"` is still present
- a top-level `runtime_cert` is still present
- `progress.jsonl` still reaches `worker_done`

## Real Continuity Benchmark

Authoritative continuity root remains:

- `tmp/iter35_kwant_walltime`

I retired the stale pre-patch attempt and relaunched the continuity root on product commit `40eec77`.

Fresh live local driver:

```text
PID 426185
.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh real cluster jobs:

- Kwant reference: `60200` on `n019`
- native RGF: `60201` on `n020`

This remains the real runtime path required by the charter:

- remote PBS `qsub`
- native `mpirun`
- no fork launchstyle for actual computation

### Kwant Reference Branch

Fresh local log from the relaunched run:

```text
[wtec][runtime] start 2026-03-13T16:03:48+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=27 load_max=10985 counts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 5, 5, 5] first_wave_thicknesses=[3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 9, 11, 13]
[kwant-bench][rank=0] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=8] start thickness_uc=5 energy_abs_ev=13.404600
[kwant-bench][rank=10] start thickness_uc=7 energy_abs_ev=13.604600
[kwant-bench][rank=13] start thickness_uc=9 energy_abs_ev=13.604600
[kwant-bench][rank=14] start thickness_uc=11 energy_abs_ev=13.604600
[kwant-bench][rank=15] start thickness_uc=13 energy_abs_ev=13.604600
```

Fresh heartbeat evidence from the same run:

```text
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=60.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=60.0
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=60.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=60.0
[kwant-bench][rank=4] heartbeat thickness_uc=3 energy_abs_ev=13.804600 elapsed_s=60.0
```

Current local checkpoint state is still flat:

```json
{
  "results_len": 5,
  "status": "partial",
  "task_count_completed": 5,
  "task_count_expected": 35
}
```

Current local shard visibility:

- `kwant_reference.rank0.jsonl`: present
- `kwant_reference.rank1.jsonl`: absent
- `kwant_reference.rank2.jsonl`: absent
- `kwant_reference.rank3.jsonl`: absent

Direct remote probe from this developer shell confirmed the fresh job assignment and current node picture:

```text
Job Id: 60200
job_state = R
exec_host = n019/64
```

```text
n019.hpc
Mem: 377 total / 78 used / 298 free GiB
16 live python3 -m wtec.transport.kwant_nanowire_benchmark ranks
several ranks at ~100% CPU with RSS around 15.1 GiB
others at ~100% CPU with RSS between ~0.2 and ~0.65 GiB
```

Interpretation:

- the thin-group fanout scheduler is live on the real cluster path
- the first live wave now spends the extra ranks inside the lightest thickness groups instead of idling them
- the run is materially active on the cluster
- but the local authority state is still flat: no fresh local completion row beyond `5/35`, no new shard files, and no benchmark-wide aggregation outputs yet

### Native-RGF Branch

The relaunched root immediately reused cached native results through:

- the full thickness-1 slice
- the full thickness-3 slice
- the full thickness-5 slice
- `d07_em0p2`
- `d07_em0p1`

Current trace tail:

```json
{"event":"rgf_case_done","job_id":null,"tag":"d07_em0p2"}
{"event":"rgf_case_start","tag":"d07_em0p1"}
{"event":"rgf_case_before_stage_transport","tag":"d07_em0p1"}
{"event":"rgf_case_done","job_id":null,"tag":"d07_em0p1"}
{"event":"rgf_case_start","tag":"d07_e0p0"}
{"event":"rgf_case_before_stage_transport","tag":"d07_e0p0"}
```

Fresh real native job:

- `60201` on `n020`

Current local artifact state for `d07_e0p0`:

- `sigma_manifest.json`: present
- `transport_result.json`: absent
- `transport_runtime_cert.json`: absent

Current synced live log for `d07_e0p0`:

```text
[wtec][runtime] start 2026-03-13T16:03:51+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T16:03:51+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=7 energy_ev=13.6046 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=8736 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
```

Interpretation:

- the native-RGF transport path remains active on the real cluster route
- the native branch is still progressing and is not the acceptance blocker
- the current blocker remains benchmark-wide completion on the Kwant / aggregation side

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
- no fresh local `done` line observed yet in `model_b/c/kwant/wtec_job.log`

## What Remains Unresolved

- benchmark-wide parity against the completed Kwant reference path
- raw/fit tolerance compliance via `comparison_raw.json` and `comparison_fit.json`
- final `>=5x` wall-time speedup proof
- separate restart acceptance; the last verified blocker remains remote SIESTA preparation failing on missing ScaLAPACK

## What The Verifier Should Do Next

1. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`; do not reset the evidence trail.
2. Verify that product commit `40eec77` is present before using this handoff.
3. Poll for the first fresh thin-group-fanout Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
4. Poll the native side until `d07_e0p0` writes:
   - `transport_result.json`
   - `transport_runtime_cert.json`
5. Do not clear acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
6. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
