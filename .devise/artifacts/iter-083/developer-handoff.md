# Iteration 83 Developer Handoff

## Summary

- No additional product code was retained this iteration.
- Branch baseline remains the weighted Kwant scheduler at commit `a5b37624f66a6051bfcdd7d740afca1c4837862b`.
- This pass refreshed evidence on the authoritative continuity root and captured a new real-cluster utilization snapshot for the patched Kwant run.

Current local `HEAD` before this handoff commit:

- `9b7b76b6fc105c422135aaa1bdca933e7479627e` (`Record iteration 81 kwant scheduler evidence`)

## Regression Surface

Focused checks rerun on current HEAD:

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
6 passed in 0.62s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.17s
```

Declared dry-test contract rerun on current HEAD:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.39s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.27s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.19s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.22s
```

Interpretation:

- the full declared regression contract is still green on current HEAD
- no new product regression was introduced by the continuity rerun work

## Local Full-Finite Parity

The last rerun of the checked-in toy full-finite workflow remains valid for this baseline:

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

Current live local driver:

```text
PID 376065
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Current live jobs:

- Kwant reference: `60175` on `n019`
- native RGF: `60176` on `n021`

The root was not replaced in this pass; I kept the authoritative continuity root and only repolled it.

## Kwant Branch

### Local checkpoint status

Local Kwant checkpoint still remains:

```json
{
  "status": "partial",
  "task_count_completed": 5,
  "task_count_expected": 35,
  "results_len": 5
}
```

Local shard visibility still remains:

- `kwant_reference.rank0.jsonl`: present
- `kwant_reference.rank1.jsonl`: absent
- `kwant_reference.rank2.jsonl`: absent
- `kwant_reference.rank3.jsonl`: absent

### Live runtime evidence

The weighted scheduler is still the live real-cluster baseline on the widened `16x4` layout:

```text
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=979 load_max=2197 counts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 7, 3]
[kwant-bench][rank=0] start thickness_uc=13 energy_abs_ev=13.604600
[kwant-bench][rank=12] start thickness_uc=5 energy_abs_ev=13.604600
[kwant-bench][rank=14] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=15] start thickness_uc=7 energy_abs_ev=13.604600
```

Latest local heartbeat evidence shows the run is still live but has not yet landed a fresh completion row:

```text
[kwant-bench][rank=14] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=620.0
[kwant-bench][rank=10] heartbeat thickness_uc=7 energy_abs_ev=13.404600 elapsed_s=360.0
[kwant-bench][rank=13] heartbeat thickness_uc=5 energy_abs_ev=13.504600 elapsed_s=520.0
[kwant-bench][rank=12] heartbeat thickness_uc=5 energy_abs_ev=13.604600 elapsed_s=520.0
[kwant-bench][rank=11] heartbeat thickness_uc=7 energy_abs_ev=13.804600 elapsed_s=360.0
[kwant-bench][rank=15] heartbeat thickness_uc=7 energy_abs_ev=13.604600 elapsed_s=340.0
```

Interpretation:

- the weighted scheduler remains active on the real PBS `qsub` + native `mpirun` path
- the first-wave task mix is still distributed across heavy and lighter thicknesses, not the old simple strided split
- however the local authority checkpoint is still flat at `5/35`, and no fresh local `done` line or new shard file appeared during this pass

### Fresh remote utilization probe

I captured a new direct remote process sample from `n019` for the current weighted-scheduler run.

Observed active rank processes:

```text
pid=354494..354503  nlwp=8  cpu≈99%   pmem≈0.6-0.7
pid=354504..354505  nlwp=8  cpu≈218-220% pmem≈13.8-14.0
pid=354506..354507  nlwp=8  cpu≈318-320% pmem≈10.1
pid=354508         nlwp=8  cpu≈170%  pmem≈3.6
```

Remote node memory snapshot:

```text
Mem total: 377 GiB
Mem used : 226 GiB
Mem free : 150 GiB
nproc    : 64
```

Interpretation:

- the patched Kwant run is doing real work on `n019`
- the node is not idle, but effective CPU use still appears materially below full 64-core occupancy while memory use is already substantial
- this now looks more like long real compute with memory/concurrency pressure than a missing local sync path

## Native-RGF Branch

The native side is still not the dominant blocker.

Retained local continuity evidence now includes the first thickness-7 point:

```json
{
  "tag": "d07_em0p2",
  "G_mean": 25.99464429036398,
  "wall_seconds": 123.842437,
  "effective_thread_count": 51.791766,
  "queue": "g4",
  "mpi_size": 1,
  "omp_threads": 64,
  "mode": "full_finite",
  "full_finite_sigma_source": "kwant_exact"
}
```

Current live native frontier is still `d07_em0p1`.

Current local artifact state:

- `transport_result.json`: absent
- `transport_runtime_cert.json`: absent
- `sigma_manifest.json`: absent

Latest local runtime log:

```text
[wtec][runtime] start 2026-03-13T15:09:21+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T15:09:21+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=7 energy_ev=13.5046 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=8736 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
[wtec][sigma] selfenergy_left_done wall_seconds=414.756 solver=lopez_sancho iterations=22
[wtec][sigma] selfenergy_right_start lead_dim=8736 solver=lopez_sancho
```

Interpretation:

- actual heavy native transport computation is still occurring on the real cluster path
- the native frontier is advancing through thickness 7 and is no longer the acceptance blocker

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

Meaning:

- benchmark-wide parity is still unproven
- raw/fit tolerance compliance is still unproven
- the required `>=5x` wall-time speedup remains unproven

## Restart Status

I did **not** rerun `wtec init` in this developer pass.

Last verified restart status remains unchanged from the prior debugger rerun:

- optional warning:
  - `meson-python: error: meson executable "meson" not found`
- material blocker:
  - `Could NOT find CustomScalapack (missing: SCALAPACK_LIBRARY)`
  - `MPI was requested, but ScaLAPACK could not be found by CMake`

Interpretation:

- restart acceptance remains independently blocked by the same remote SIESTA / ScaLAPACK defect

## What The Verifier Should Do Next

1. Verify the current branch head that records this refreshed continuity evidence.
2. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`.
3. Continue polling for the first fresh Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
4. Continue polling the native side until `d07_em0p1` returns its local artifact set.
5. Do not claim acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
6. If more runtime tuning is attempted later, use the new remote utilization evidence here:
   - real compute is active
   - local sync is not the current blocker
   - the remaining question is whether further Kwant concurrency tuning can improve throughput without exhausting node memory
7. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
