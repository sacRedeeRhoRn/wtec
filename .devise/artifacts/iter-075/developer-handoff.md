# Iteration 75 Developer Handoff

## What Changed

- Patched `wtec/transport/nanowire_benchmark_cluster.py` so the default Kwant worker layout now favors task-level parallelism on a 64-core node:
  - old default: `mpi_np=4`, `omp_threads=16`
  - new default: `mpi_np=16`, `omp_threads=4`
- Kept `TOPOSLAB_KWANT_BENCH_MPI_RANKS` as the explicit escape hatch; overrides still win if the operator wants a different geometry.
- Updated the focused layout/walltime expectations in `tests/test_nanowire_benchmark_cluster.py`:
  - staged Kwant PBS script now expects `mpiprocs=16:ompthreads=4`
  - derived default Kwant walltime now expects `03:00:00` instead of `09:00:00`

## Why This Change Was Necessary

- Direct remote inspection of the authoritative continuity run showed the prior `4x16` Kwant layout was underutilizing the node on the real cluster path.
- On node `n022`, the old job had only four active worker ranks, each consuming about `376%` to `401%` CPU while the rest of the 64-core node remained largely idle.
- That meant the current blocker was no longer “missing logs” or “missing shard retrieval”; it was a real execution-layout defect on the lagging Kwant reference branch.
- The benchmark points are independent, so a higher-rank / lower-thread default is the durable fix: it increases concurrent Kwant reference points without changing the native-RGF execution contract.

## Tests Passed

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

### Old continuity diagnosis

Before the patch, direct remote inspection of the authoritative continuity run showed the prior `4x16` Kwant layout underutilizing the node:

```text
377535     526  401  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377536     526  399  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377537     526  376  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377538     526  376  3.6 python3 -m wtec.transport.kwant_nanowire_benchmark ...
```

That was the decisive evidence for changing the default worker layout.

### Fresh authoritative relaunch on the new layout

I retired the old `4x16` continuity attempt after the native cache was safely through `d05_em0p1`, then relaunched the same authoritative root:

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
PID 333802
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh remote jobs at handoff:

```text
60152  kwc_w13_l24      R
60153  rgf_prima_nanow  R
```

The fresh staged Kwant PBS script proves the new default geometry is active on the real path:

```text
#PBS -l select=1:ncpus=64:mpiprocs=16:ompthreads=4
#PBS -l walltime=03:00:00
...
mpirun -np 16 --bind-to none ... python3 -m wtec.transport.kwant_nanowire_benchmark ...
```

The unique remote workdir identity is still active:

```text
/home/msj/Desktop/playground/electroics/wtec/remote_runs/nanowire_benchmark/mp-1018028/iter35_kwant_walltime_model_b_c_kwant_11354363f4
```

The fresh remote directory still contains the authoritative staged resume state:

```text
TiS_model_b_c_c_canonical_hr.dat
kwant_payload.json
kwant_reference.json
kwant_reference.pbs
kwant_reference.rank0.jsonl
wtec_job.log
wtec_src.zip
```

The live local Kwant log now shows the widened 16-rank fan-out across multiple thickness waves instead of only the first four thickness-3 points:

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
[kwant-bench][rank=0] start thickness_uc=3 energy_abs_ev=13.404600
[kwant-bench][rank=1] start thickness_uc=3 energy_abs_ev=13.504600
[kwant-bench][rank=2] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=3] start thickness_uc=3 energy_abs_ev=13.704600
[kwant-bench][rank=5] start thickness_uc=5 energy_abs_ev=13.404600
[kwant-bench][rank=6] start thickness_uc=5 energy_abs_ev=13.504600
[kwant-bench][rank=7] start thickness_uc=5 energy_abs_ev=13.604600
```

The first heartbeat wave on the new run is also visible locally:

```text
[kwant-bench][rank=1] heartbeat thickness_uc=3 energy_abs_ev=13.504600 elapsed_s=20.0
[kwant-bench][rank=4] heartbeat thickness_uc=3 energy_abs_ev=13.804600 elapsed_s=20.0
[kwant-bench][rank=3] heartbeat thickness_uc=3 energy_abs_ev=13.704600 elapsed_s=20.0
[kwant-bench][rank=2] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=20.0
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.404600 elapsed_s=20.0
```

Fresh direct remote process inspection on `n022` confirms the wider worker fan-out is materialized, not merely staged:

```text
377949      65  0.0  0.0 /bin/sh ... mpirun -np 16 ...
377954      65  0.0  0.0 mpiexec.hydra -np 16 ...
377977      65  158  2.3 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377978      65  174  2.3 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377979      65  134  2.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377980      65  173  2.3 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377981      65  168  2.3 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377982      65 99.0  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377983      65 99.0  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377984      65 99.0  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377985      65 99.1  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377986      65 99.1  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377987      65 99.0  0.2 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377988      65 99.0  0.2 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377989      65 99.0  0.2 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377990      65 98.9  0.2 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377991      65 99.0  0.2 python3 -m wtec.transport.kwant_nanowire_benchmark ...
377992      65 98.8  0.1 python3 -m wtec.transport.kwant_nanowire_benchmark ...
```

### Native side continuity state

The native side remains on the real PBS `qsub` + native `mpirun` path and is no longer the dominant blocker.

The fresh relaunch reused:

- the full thickness-1 slice
- the full thickness-3 slice
- `d05_em0p2`
- `d05_em0p1`

Current continuity trace proves the live frontier is now `d05_e0p0`:

```text
{"event":"rgf_case_done","job_id":null,"tag":"d03_e0p2"}
{"event":"rgf_case_done","job_id":null,"tag":"d05_em0p2"}
{"event":"rgf_case_done","job_id":null,"tag":"d05_em0p1"}
{"event":"rgf_case_start","tag":"d05_e0p0"}
{"event":"rgf_case_before_stage_transport","tag":"d05_e0p0"}
```

The current live `d05_e0p0` runtime log shows the job is active on the real cluster path:

```text
[wtec][runtime] start 2026-03-13T14:19:34+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T14:19:34+09:00
[wtec][sigma] full_finite_principal_start hr_path=TiS_model_b_c_c_canonical_hr.dat length_uc=24 width_uc=13 thickness_uc=5 energy_ev=13.6046 eta_ev=1e-06
[wtec][sigma] full_finite_principal_geometry_ready norb=16 principal_layer_width=6 pad_x=5 nx_effective=34 lead_dim=6240 slice_widths=6,6,6,10,6
[wtec][sigma] full_finite_principal_blocks_ready h_lead_shape=6240x6240 v_lead_shape=6240x6240
[wtec][sigma] selfenergy_left_start lead_dim=6240 solver=lopez_sancho
```

## Still Unresolved

- The local merged Kwant checkpoint is still partial at `5/35`.
- Only `kwant_reference.rank0.jsonl` is currently visible locally; `rank1` through `rank3` remain absent.
- Benchmark-wide aggregation files are still absent:
  - `tmp/iter35_kwant_walltime/model_b/c/rgf/rgf_raw.json`
  - `tmp/iter35_kwant_walltime/model_b/c/comparison_raw.json`
  - `tmp/iter35_kwant_walltime/model_b/c/comparison_fit.json`
  - `tmp/iter35_kwant_walltime/benchmark_summary.json`
- Because those authority files are still absent, benchmark-wide parity, raw/fit tolerance compliance, and the required `>=5x` wall-time speedup remain unproven.
- I did not rerun `wtec init` after this layout patch. The last verified restart blocker remains the separate remote SIESTA / ScaLAPACK failure, with the earlier optional `python-mumps` / `meson` warning still present.

## What The Verifier Should Do Next

1. Verify the current developer `HEAD` on a clean checkout of this branch.
2. Re-run the declared dry-test contract and the focused layout regressions:
   - `tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or ... or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"`
   - `tests/test_kwant_nanowire_benchmark_resume.py`
3. Confirm the fresh continuity root now stages and runs the Kwant branch with:
   - `mpiprocs=16`
   - `ompthreads=4`
   - `walltime=03:00:00`
4. Confirm the live continuity log still shows the widened 16-rank fan-out across thicknesses `3`, `5`, `7`, and `9`.
5. Continue polling the authoritative continuity root until one of these happens:
   - the local Kwant checkpoint advances beyond `5/35`
   - new local shard files `kwant_reference.rank1.jsonl` / `rank2` / `rank3` appear
   - `d05_e0p0` returns its final local artifact set
   - the benchmark writes `rgf_raw.json`, `comparison_raw.json`, `comparison_fit.json`, and `benchmark_summary.json`
6. Keep the ScaLAPACK / SIESTA restart defect separate from the benchmark continuity verification.
