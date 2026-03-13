# Iteration 93 Developer Handoff

## What Changed

- No new product code was retained this iteration.
- Current branch HEAD before this handoff commit: `02860c4e601d2b6800136c1cdbf3088153d19461`
- The live benchmark continues on the existing thin-group fanout baseline from product commit `40eec77fe36760623f6b595ab50b7d889bd53c42`.

Reasoning:

- I used fresh direct remote probes from this developer shell to answer the open question from the last pass.
- The current blocker is not a local shard-retrieval bug: the remote Kwant workdir itself still only contains the seeded `kwant_reference.rank0.jsonl`.
- That means the current missing benchmark-wide artifacts still come from real compute completion not landing yet, not from local live-sync dropping finished shard files.

## Regression And Parity Status

No product code changed from the previously verified baseline. Current code state therefore remains the same one already re-verified green on current HEAD in the debugger pass:

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
7 passed in 0.56s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.18s

.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.11s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.17s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.16s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.17s
```

The checked-in toy full-finite parity workflow also remains green on the unchanged code baseline:

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

Current qstat state:

```text
60200 R
60201 R
```

This remains the real chartered path:

- remote PBS `qsub`
- native `mpirun`
- no fork launchstyle for actual computation

### Kwant Reference Branch

Local authority state remains flat:

```json
{
  "results_len": 5,
  "status": "partial",
  "task_count_completed": 5,
  "task_count_expected": 35
}
```

Local shard visibility remains:

- `kwant_reference.rank0.jsonl`: present
- `kwant_reference.rank1.jsonl`: absent
- `kwant_reference.rank2.jsonl`: absent
- `kwant_reference.rank3.jsonl`: absent

Fresh remote probe from this developer shell showed the same remote state:

```text
Job Id: 60200
job_state = R
exec_host = n019/64

remote files:
- TiS_model_b_c_c_canonical_hr.dat
- kwant_payload.json
- kwant_reference.json
- kwant_reference.pbs
- kwant_reference.rank0.jsonl
- wtec_job.log
- wtec_src.zip
```

There are still no remote `kwant_reference.rank1.jsonl`, `rank2`, or `rank3` files.

Remote log tail still shows only heartbeats, no completion lines:

```text
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=960.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=960.0
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=960.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=960.0
[kwant-bench][rank=4] heartbeat thickness_uc=3 energy_abs_ev=13.804600 elapsed_s=960.0
```

Fresh remote node snapshot for the current Kwant run:

```text
n019.hpc
Mem: 377 total / 270 used / 106 free GiB
16 live python3 -m wtec.transport.kwant_nanowire_benchmark ranks
five ranks around 40.1 GiB RSS and ~290-298% CPU
five ranks around 14.6 GiB RSS and ~206-213% CPU
three ranks around 1.7-2.9 GiB RSS and ~98% CPU
three defunct helper processes also visible
```

Interpretation:

- the thin-group fanout run is real and materially active on the cluster
- the node is heavily occupied
- the current problem is still throughput to first completion, not missing local retrieval of already-written remote shard files

### Native-RGF Branch

The native side remains active and ahead of the Kwant side.

Current `d07_e0p0` local artifact state:

- `sigma_manifest.json`: present
- `transport_result.json`: absent
- `transport_runtime_cert.json`: absent

Current synced local log:

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

Fresh remote probe for job `60201`:

```text
Job Id: 60201
job_state = R
exec_host = n020/64

n020.hpc
Mem: 377 total / 18 used / 317 free GiB
python3 -m wtec.transport.kwant_sigma_extract ... thickness_uc=7 ... at ~5719% CPU with 127 threads and RSS ~17.6 GiB
```

Interpretation:

- actual heavy native-RGF transport computation continues on the real cluster path
- the native branch is not the acceptance blocker
- the current bottleneck remains on the Kwant reference side

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
- no fresh remote `done` line observed either

## What Remains Unresolved

- benchmark-wide parity against the completed Kwant reference path
- raw/fit tolerance compliance via `comparison_raw.json` and `comparison_fit.json`
- final `>=5x` wall-time speedup proof
- separate restart acceptance; the last verified blocker remains remote SIESTA preparation failing on missing ScaLAPACK

## What The Verifier Should Do Next

1. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`; do not reset the evidence trail.
2. Verify current HEAD before using this handoff.
3. Continue polling for the first fresh Kwant completion signal on both local and remote views:
   - local `kwant_reference.json` advancing beyond `5/35`
   - appearance of local or remote `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in either local or remote `wtec_job.log`
4. Continue polling the native side until `d07_e0p0` writes:
   - `transport_result.json`
   - `transport_runtime_cert.json`
5. Do not clear acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
6. Treat the current scheduler question as a throughput-versus-memory tradeoff:
   - the run is real
   - the remote node is heavily occupied
   - the present evidence does not support calling the remaining issue a sync bug
7. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
