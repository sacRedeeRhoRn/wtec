# Iteration 77 Developer Handoff

## What Changed This Iteration

- No new product code was retained beyond the current branch `HEAD`.
- This developer pass focused on validating the already-landed widened Kwant layout on the authoritative continuity root and capturing fresh real-runtime evidence after the relaunch.
- I wrote this iteration handoff so the debugger can continue from the current live frontier without re-deriving the runtime state.

## Current Code Baseline

Current branch `HEAD` already contains the retained Kwant-layout change:

- default Kwant benchmark layout widened from `4x16` to `16x4`
- staged Kwant walltime correspondingly reduced from `09:00:00` to `03:00:00`
- resume staging, unique remote workdir identity, and live heartbeat plumbing are still present

Current `HEAD` commit at handoff time:

```text
f440727c16d7efa0ecaefb205ecddefedc405816
```

## Regression Status

I did not change product code again in this pass, but I revalidated the already-landed layout patch earlier in the same developer cycle, and the following remain green on current `HEAD`:

Focused regressions:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.24s

.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
4 passed in 0.58s
```

Declared dry-test contract:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.81s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.24s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.23s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.27s
```

Local toy full-finite parity also remains green:

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

## Real Continuity Benchmark State

Authoritative continuity root remains:

```text
tmp/iter35_kwant_walltime
```

Current live local driver:

```text
PID 333802
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Current live remote jobs:

```text
60152  kwc_w13_l24      R
60153  rgf_prima_nanow  R
```

### Kwant branch

The staged PBS contract remains the widened real runtime geometry:

```text
#PBS -l select=1:ncpus=64:mpiprocs=16:ompthreads=4
#PBS -l walltime=03:00:00
mpirun -np 16 --bind-to none ... python3 -m wtec.transport.kwant_nanowire_benchmark ...
```

The unique remote Kwant workdir remains active:

```text
/home/msj/Desktop/playground/electroics/wtec/remote_runs/nanowire_benchmark/mp-1018028/iter35_kwant_walltime_model_b_c_kwant_11354363f4
```

The remote directory still contains the staged authoritative resume state:

```text
TiS_model_b_c_c_canonical_hr.dat
kwant_payload.json
kwant_reference.json
kwant_reference.pbs
kwant_reference.rank0.jsonl
wtec_job.log
wtec_src.zip
```

Current local merged Kwant checkpoint is still:

```json
{
  "status": "partial",
  "task_count_completed": 5,
  "task_count_expected": 35,
  "results_len": 5
}
```

Current local shard visibility is still:

- `kwant_reference.rank0.jsonl`: present
- `kwant_reference.rank1.jsonl`: absent
- `kwant_reference.rank2.jsonl`: absent
- `kwant_reference.rank3.jsonl`: absent

Fresh local Kwant log evidence now shows the widened fan-out is materially alive across multiple thickness waves:

```text
[wtec][runtime] start 2026-03-13T14:19:31+09:00
[kwant-bench][rank=4] start thickness_uc=3 energy_abs_ev=13.804600
[kwant-bench][rank=8] start thickness_uc=5 energy_abs_ev=13.704600
[kwant-bench][rank=9] start thickness_uc=5 energy_abs_ev=13.804600
[kwant-bench][rank=10] start thickness_uc=7 energy_abs_ev=13.404600
[kwant-bench][rank=11] start thickness_uc=7 energy_abs_ev=13.504600
[kwant-bench][rank=12] start thickness_uc=7 energy_abs_ev=13.604600
[kwant-bench][rank=13] start thickness_uc=7 energy_abs_ev=13.704600
[kwant-bench][rank=14] start thickness_uc=7 energy_abs_ev=13.804600
[kwant-bench][rank=15] start thickness_uc=9 energy_abs_ev=13.404600
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
```

and the current heartbeat wave spans both thickness-3 and thickness-5 ranks:

```text
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=700.0
[kwant-bench][rank=4] heartbeat thickness_uc=3 energy_abs_ev=13.804600 elapsed_s=700.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=700.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=700.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=700.0
[kwant-bench][rank=5] heartbeat thickness_uc=5 energy_abs_ev=13.404600 elapsed_s=621.0
[kwant-bench][rank=9] heartbeat thickness_uc=5 energy_abs_ev=13.804600 elapsed_s=621.0
[kwant-bench][rank=7] heartbeat thickness_uc=5 energy_abs_ev=13.604600 elapsed_s=620.5
[kwant-bench][rank=12] heartbeat thickness_uc=7 energy_abs_ev=13.604600 elapsed_s=440.0
[kwant-bench][rank=8] heartbeat thickness_uc=5 energy_abs_ev=13.704600 elapsed_s=620.1
[kwant-bench][rank=6] heartbeat thickness_uc=5 energy_abs_ev=13.504600 elapsed_s=620.0
[kwant-bench][rank=14] heartbeat thickness_uc=7 energy_abs_ev=13.804600 elapsed_s=440.0
[kwant-bench][rank=13] heartbeat thickness_uc=7 energy_abs_ev=13.704600 elapsed_s=440.0
```

Fresh direct remote process inspection on `n022` confirms the widened layout is materially active, not just staged:

```text
377949     742  0.0  0.0 /bin/sh ... mpirun -np 16 ...
377954     742  0.0  0.0 mpiexec.hydra -np 16 ...
377977     742  182  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377978     742  175  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377979     742  189  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377980     742  190  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377981     742  176  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377982     742  322 10.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377983     742  317 10.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377984     742  318 10.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377985     742  317 10.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377986     742  322 10.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377992     742 98.6  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
```

Interpretation:

- the widened `16x4` Kwant layout is live and materially using more concurrent task-level parallelism than the earlier underutilized layout
- the current blocker is still computation progress/completion on the full Kwant reference sweep, not missing observability or directory-collision plumbing
- but no new completed local checkpoint rows have been retrieved yet beyond the original `5/35`

### Native-RGF branch

The native branch is still not the dominant blocker.

The continuity trace proves the relaunch reused:

- the full thickness-1 slice
- the full thickness-3 slice
- `d05_em0p2`
- `d05_em0p1`

and has already advanced the live frontier through `d05_e0p0` into `d05_e0p1`:

```text
{"event":"rgf_case_done","job_id":null,"tag":"d05_em0p2"}
{"event":"rgf_case_done","job_id":null,"tag":"d05_em0p1"}
{"event":"rgf_case_done","job_id":"60153","tag":"d05_e0p0"}
{"event":"rgf_case_start","tag":"d05_e0p1"}
{"event":"rgf_case_before_stage_transport","tag":"d05_e0p1"}
```

The current live native frontier `d05_e0p1` has not yet returned its final local artifact set:

```text
transport_result.json         missing
transport_runtime_cert.json   missing
sigma_manifest.json           missing
```

The current live `d05_e0p1` log remains on the real cluster path and has crossed the first Lopez-Sancho self-energy boundary:

```text
[wtec][runtime] start 2026-03-13T14:27:13+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T14:27:13+09:00
[wtec][sigma] full_finite_principal_start ... thickness_uc=5 energy_ev=13.7046 ...
[wtec][sigma] full_finite_principal_geometry_ready ... lead_dim=6240 ...
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=6240x6240 v_lead_shape=6240x6240
[wtec][sigma] selfenergy_left_start lead_dim=6240 solver=lopez_sancho
```

## Still Unresolved

- The local Kwant merged checkpoint is still partial at `5/35`.
- Only `kwant_reference.rank0.jsonl` is visible locally.
- Benchmark-wide authority outputs are still absent:
  - `tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json`
  - `tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json`
  - `tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json`
  - `tmp/iter35_kwant_walltime/benchmark_summary.json`
- Because those files are still missing, benchmark-wide parity, raw/fit tolerance compliance, and the required `>=5x` speedup remain unproven.
- I did not rerun `wtec init` in this developer pass. The last verified restart blocker remains the separate remote SIESTA / ScaLAPACK failure, with the earlier optional `python-mumps` / `meson` warning still present.

## What The Verifier Should Do Next

1. Verify the current developer `HEAD` on a clean checkout of this branch.
2. Re-run the declared dry-test contract and the focused layout regressions if you need to re-establish the code baseline.
3. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`.
4. Confirm the widened Kwant layout is still live on the real path:
   - `mpiprocs=16`
   - `ompthreads=4`
   - `walltime=03:00:00`
   - multi-thickness fan-out in the live Kwant log
5. Continue polling until one of these happens locally:
   - `kwant_reference.json` advances beyond `5/35`
   - `kwant_reference.rank1.jsonl`, `rank2`, or `rank3` appear
   - `d05_e0p1` returns its final local artifact set
   - the benchmark writes `rgf_raw.json`, `comparison_raw.json`, `comparison_fit.json`, and `benchmark_summary.json`
6. Keep the ScaLAPACK / SIESTA restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
