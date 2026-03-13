# Iteration 73 Developer Handoff

## What Changed

- Patched `wtec/transport/nanowire_benchmark_cluster.py` so the remote Kwant benchmark workdir is derived from the full local benchmark root identity instead of only the trailing `model_b/c/kwant` path. The new path builder keeps the readable local tail and appends a stable SHA1 suffix.
- Patched `wtec/transport/nanowire_benchmark_cluster.py` so every resumed Kwant submission stages the local `kwant_reference.json` checkpoint and any `kwant_reference.rank*.jsonl` shard files into the remote workdir before launch.
- Added focused coverage in `tests/test_nanowire_benchmark_cluster.py` for both contracts:
  - different local benchmark roots map to different remote Kwant workdirs
  - resumed submissions stage the local partial checkpoint and shard files

## Why This Change Was Necessary

- The prior remote-dir scheme only used the last three local path components, so different local continuity/probe roots that both ended in `model_b/c/kwant` collided on the same remote Kwant directory.
- That collision made it possible for an overlapping probe to interfere with the authoritative continuity run.
- Even after local resume support existed, fresh remote reruns could still start without the authoritative local partial checkpoint/shard state unless those files were explicitly staged into the new remote directory.

## Tests Passed

Focused regressions:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards"
7 passed, 3 deselected in 0.23s

.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
4 passed in 0.56s
```

Declared dry-test contract:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.68s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.21s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.17s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.18s
```

Local toy full-finite parity was also refreshed after rebuilding the native runner:

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

## Real Runtime Evidence

Fresh authoritative continuity relaunch:

```bash
export TOPOSLAB_KWANT_BENCH_HEARTBEAT_SECONDS=20
.venv/bin/wtec benchmark-transport \
  examples/sio2_tap_sio2_small/run_small.json \
  --output-dir tmp/iter35_kwant_walltime \
  --queue g4 \
  --walltime 01:00:00
```

Current local driver at handoff:

```text
PID 322475
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh remote jobs at handoff:

```text
60143  kwc_w13_l24      R
60144  rgf_prima_nanow  R
```

The fresh staged Kwant PBS script proves the new unique remote workdir is active on the real path:

```text
/home/msj/Desktop/playground/electroics/wtec/remote_runs/nanowire_benchmark/mp-1018028/iter35_kwant_walltime_model_b_c_kwant_11354363f4
```

The staged script still satisfies the runtime contract:

- real PBS `qsub`
- native `mpirun`
- no fork launchstyle
- heartbeat env override forwarded

Fresh direct remote probe of that unique workdir confirmed the resumed local checkpoint state was staged into the new remote directory:

```text
TiS_model_b_c_c_canonical_hr.dat
kwant_payload.json
kwant_reference.json
kwant_reference.pbs
kwant_reference.rank0.jsonl
wtec_job.log
wtec_src.zip
```

Current local Kwant state remains the lagging side, but it is now isolated to the authoritative remote directory and continues to emit live per-rank heartbeats:

```json
{
  "status": "partial",
  "task_count_completed": 5,
  "task_count_expected": 35,
  "results_len": 5
}
```

```text
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=20.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=20.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=20.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=20.0
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=100.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=100.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=100.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=100.0
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=180.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=180.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=180.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=180.0
```

The native side reused the full cached thickness-1, thickness-3, and `d05_em0p2` results locally and advanced the live frontier into `d05_em0p1`:

```text
{"event":"rgf_case_done","job_id":null,"tag":"d03_em0p2"}
{"event":"rgf_case_done","job_id":null,"tag":"d03_em0p1"}
{"event":"rgf_case_done","job_id":null,"tag":"d03_e0p0"}
{"event":"rgf_case_done","job_id":null,"tag":"d03_e0p1"}
{"event":"rgf_case_done","job_id":null,"tag":"d03_e0p2"}
{"event":"rgf_case_done","job_id":null,"tag":"d05_em0p2"}
{"event":"rgf_case_start","tag":"d05_em0p1"}
{"event":"rgf_case_before_stage_transport","tag":"d05_em0p1"}
```

`d05_em0p2` is already retrieved locally on this continuity root. Representative value:

```json
{
  "G_mean": [17.99734029428402]
}
```

The current live native frontier `d05_em0p1` is staged and running on the real cluster path, but it has not yet returned a final local artifact set:

```text
[wtec][runtime] start 2026-03-13T14:07:34+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T14:07:34+09:00
[wtec][sigma] full_finite_principal_start hr_path=TiS_model_b_c_c_canonical_hr.dat length_uc=24 width_uc=13 thickness_uc=5 energy_ev=13.5046 eta_ev=1e-06
[wtec][sigma] full_finite_principal_geometry_ready norb=16 principal_layer_width=6 pad_x=5 nx_effective=34 lead_dim=6240 slice_widths=6,6,6,10,6
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=6240x6240 v_lead_shape=6240x6240
[wtec][sigma] selfenergy_left_start lead_dim=6240 solver=lopez_sancho
[wtec][sigma] selfenergy_left_done wall_seconds=176.242 solver=lopez_sancho iterations=22
[wtec][sigma] selfenergy_right_start lead_dim=6240 solver=lopez_sancho
```

## Still Unresolved

- The Kwant reference branch remains locally partial at `5/35`.
- Only `kwant_reference.rank0.jsonl` is currently visible locally; `rank1` through `rank3` are still absent.
- Benchmark-wide aggregation files are still absent:
  - `tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json`
  - `tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json`
  - `tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json`
  - `tmp/iter35_kwant_walltime/benchmark_summary.json`
- Because those authority files are still absent, benchmark-wide parity, raw/fit tolerance compliance, and the required `>=5x` wall-time speedup remain unproven.
- The restart-path defect remains separate and unchanged: remote SIESTA preparation still fails on missing ScaLAPACK, with the earlier optional `python-mumps` / `meson` warning still present.

## What The Verifier Should Do Next

1. Verify the current developer `HEAD` on a clean checkout of this branch.
2. Re-run the declared dry-test contract and the focused new regressions:
   - `tests/test_nanowire_benchmark_cluster.py -k "... kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards"`
   - `tests/test_kwant_nanowire_benchmark_resume.py`
3. Confirm the fresh continuity root still uses the unique remote Kwant workdir:
   - `iter35_kwant_walltime_model_b_c_kwant_11354363f4`
4. Confirm the new remote Kwant workdir contains the staged resume files:
   - `kwant_reference.json`
   - `kwant_reference.rank0.jsonl`
5. Continue polling the authoritative continuity root until one of these happens:
   - the Kwant checkpoint advances beyond `5/35`
   - new local shard files `kwant_reference.rank1.jsonl` / `rank2` / `rank3` appear
   - `d05_em0p1` returns its final local artifact set
   - the benchmark writes `rgf_raw.json`, `comparison_raw.json`, `comparison_fit.json`, and `benchmark_summary.json`
6. Keep the ScaLAPACK / SIESTA restart defect separate from the benchmark continuity verification.
