# Iteration 97 Developer Handoff

## What Changed

- Product commit: `68b287844b8c4768d73ecb8380e06863f014aa31` (`Default kwant benchmark to grouped thicknesses`)
- Current handoff commit target before this file is committed: `68b287844b8c4768d73ecb8380e06863f014aa31`

Changed files this iteration:

- `wtec/transport/kwant_nanowire_benchmark.py`
- `tests/test_kwant_nanowire_benchmark_resume.py`

Behavior change:

- restored one-rank-per-thickness as the default Kwant benchmark scheduler when `world_size >= number_of_thickness_groups`
- kept the thin-group fanout logic behind explicit opt-in:
  - `TOPOSLAB_KWANT_BENCH_SPLIT_THICKNESS_GROUPS=1`

Why this change:

- fresh remote evidence from the thin-group run showed the current real blocker was throughput under severe memory pressure, not a local sync bug
- the grouped-thickness relaunch preserves thickness-local reuse and materially reduces node memory pressure on the real cluster path

## Regression Evidence

Focused benchmark regressions passed on the patched code:

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
8 passed in 0.55s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.16s
```

Declared dry-test contract also passed:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.10s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.20s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.18s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.22s
```

## Local Full-Finite Parity

Refreshed the checked-in toy parity workflow again after the scheduler patch. Result:

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

I retired the thin-group run and relaunched the continuity root on product commit `68b2878`.

Fresh live local driver:

```text
PID 449786
.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh real cluster jobs:

- Kwant reference: `60211` on `n019`
- native RGF: `60212` on `n021`

Current qstat status:

```text
60211 R
60212 R
```

This remains the real chartered path:

- remote PBS `qsub`
- native `mpirun`
- no fork launchstyle for actual computation

### Kwant Reference Branch

Fresh local log from the relaunched grouped-thickness run:

```text
[wtec][runtime] start 2026-03-13T16:28:11+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=0 load_max=10985 counts=[5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] first_wave_thicknesses=[3, 5, 7, 9, 11, 13]
[kwant-bench][rank=0] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=1] start thickness_uc=5 energy_abs_ev=13.604600
[kwant-bench][rank=2] start thickness_uc=7 energy_abs_ev=13.604600
[kwant-bench][rank=3] start thickness_uc=9 energy_abs_ev=13.604600
[kwant-bench][rank=4] start thickness_uc=11 energy_abs_ev=13.604600
[kwant-bench][rank=5] start thickness_uc=13 energy_abs_ev=13.604600
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=60.0
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

Fresh direct remote probe from this developer shell confirmed the relaunch and materially lower node pressure:

```text
Job 60211
job_state = R
exec_host = n019/64
```

```text
n019.hpc
Mem: 377 total / 20 used / 356 free GiB
16 live python3 -m wtec.transport.kwant_nanowire_benchmark ranks
one rank around 14.6 GiB RSS at ~204% CPU
most remaining ranks between ~0.14 GiB and ~1.5 GiB RSS at ~80-81% CPU
```

Compared with the thin-group run, the grouped-thickness relaunch reduced observed node memory pressure from about `270 GiB` used to about `20 GiB` used on `n019`.

Interpretation:

- the grouped-thickness default is live on the real cluster path
- the relaunch materially reduces duplicated-memory pressure
- the current acceptance blocker is still time-to-first-Kwant-completion, not a retrieval bug
- at handoff time the local authority state is still flat: no completion beyond `5/35`, no new shard files, and no fresh local `done` line yet

### Native-RGF Branch

The relaunched root immediately reused cached native results through:

- the full thickness-1 slice
- the full thickness-3 slice
- the full thickness-5 slice
- `d07_em0p2`
- `d07_em0p1`
- `d07_e0p0`

Current trace tail:

```json
{"event":"rgf_case_done","job_id":null,"tag":"d07_em0p1"}
{"event":"rgf_case_start","tag":"d07_e0p0"}
{"event":"rgf_case_before_stage_transport","tag":"d07_e0p0"}
{"event":"rgf_case_done","job_id":null,"tag":"d07_e0p0"}
{"event":"rgf_case_start","tag":"d07_e0p1"}
{"event":"rgf_case_before_stage_transport","tag":"d07_e0p1"}
```

Fresh real native job:

- `60212` on `n021`

Current local artifact state for `d07_e0p1`:

- `sigma_manifest.json`: missing
- `transport_result.json`: missing
- `transport_runtime_cert.json`: missing

Current synced live log for `d07_e0p1`:

```text
[wtec][runtime] start 2026-03-13T16:28:14+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T16:28:14+09:00
[wtec][sigma] full_finite_principal_start hr_path=TiS_model_b_c_c_canonical_hr.dat length_uc=24 width_uc=13 thickness_uc=7 energy_ev=13.7046 eta_ev=1e-06
[wtec][sigma] full_finite_principal_geometry_ready norb=16 principal_layer_width=6 pad_x=5 nx_effective=34 lead_dim=8736 slice_widths=6,6,6,10,6
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=8736x8736 v_lead_shape=8736x8736
[wtec][sigma] selfenergy_left_start lead_dim=8736 solver=lopez_sancho
[wtec][sigma] selfenergy_left_done wall_seconds=401.156 solver=lopez_sancho iterations=22
[wtec][sigma] selfenergy_right_start lead_dim=8736 solver=lopez_sancho
```

Fresh direct remote probe for the new native job:

```text
Job 60212
job_state = R
exec_host = n021/64

n021.hpc
Mem: 377 total / 17 used / 330 free GiB
python3 -m wtec.transport.kwant_sigma_extract ... thickness_uc=7 ... at ~5627% CPU with 127 threads and RSS ~16.4 GiB
```

Interpretation:

- the native-RGF transport path remains active on the real cluster route
- the native branch is still advancing and is not the acceptance blocker
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
2. Verify that product commit `68b2878` is present before using this handoff.
3. Poll for the first fresh grouped-thickness Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
4. Poll the native side until `d07_e0p1` writes:
   - `sigma_manifest.json`
   - `transport_result.json`
   - `transport_runtime_cert.json`
5. Do not clear acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
6. Treat the current tradeoff as a throughput-versus-memory question:
   - the grouped-thickness relaunch is real
   - the node-memory picture improved materially
   - the remaining issue is still time-to-first-Kwant-completion, not a retrieval bug
7. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
