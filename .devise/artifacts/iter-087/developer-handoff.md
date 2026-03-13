# Iteration 87 Developer Handoff

## What Changed

- Product change: regroup the live Kwant benchmark sweep by thickness so each heavy finalized Kwant system is built once per rank and then reused across all requested energies for that thickness.
- Landed product commit: `ff1bced` (`Group kwant benchmark work by thickness`).
- Files changed:
  - [`wtec/transport/kwant_nanowire_benchmark.py`](../../../../wtec/transport/kwant_nanowire_benchmark.py)
  - [`tests/test_kwant_nanowire_benchmark_resume.py`](../../../../tests/test_kwant_nanowire_benchmark_resume.py)

### Runtime change

The previous lighter-first scheduler still launched many independent ranks on the same thickness, which duplicated the expensive Kwant finalized system across the node. A direct remote probe of the prior run showed the Kwant node at about `265 GiB` used with the first live wave spread across repeated thickness-3/5/7 tasks and no fresh checkpoint row landing locally.

The new distributor now:

- keeps all pending energies for a thickness on one rank when enough ranks are available
- preserves deterministic light-to-heavy local execution within each thickness group
- emits a new real-run first wave of one rank per pending thickness instead of duplicating the same thickness across many ranks

That keeps the same real PBS `qsub` + native `mpirun` execution contract, but removes the repeated in-node Kwant system duplication that was driving the earlier memory-heavy live run.

### New focused coverage

Updated the benchmark scheduler regressions to prove:

- all tasks for a thickness stay on one rank, both when ranks are plentiful and when ranks are constrained
- the first live wave on a wide MPI layout now seeds one rank per thickness group (`3, 5, 7, 9, 11, 13`)

## Focused Regressions

```text
.venv/bin/pytest -q tests/test_kwant_nanowire_benchmark_resume.py
7 passed in 0.62s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "uses_conservative_multi_rank_layout or forwards_cancel_event or resubmits_from_partial_checkpoint or resubmits_from_partial_rank_shards or forwards_heartbeat_env_override or kwant_remote_dir_includes_local_root_identity or stages_local_partial_checkpoint_and_shards or resolve_kwant_reference_walltime_scales_by_worker_waves or kwant_worker_layout_honors_env_override"
9 passed, 1 deselected in 0.21s
```

## Declared Dry-Test Contract

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
4 passed, 3 deselected in 4.48s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 9 deselected in 0.22s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.16s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
14 passed in 0.37s
```

Interpretation:

- the grouped-thickness scheduler change keeps the full declared regression contract green

## Local Full-Finite Parity

Reran the checked-in toy full-finite workflow after rebuilding the local native runner:

```bash
make -C wtec/ext/rgf clean all
.venv_verify/bin/python -m wtec.transport.kwant_sigma_extract \
  --hr-path tmp/rgf_native_smoke/toy_hr.dat \
  --length-uc 3 --width-uc 1 --thickness-uc 1 \
  --energy-ev 0.0 --eta-ev 1e-6 \
  --out-dir tmp/devise_rgf_full_finite/sigma
(cd tmp/devise_rgf_full_finite && ../../wtec/ext/rgf/build/wtec_rgf_runner full_finite_payload.json full_finite_raw.json)
```

Observed result:

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

- the checked-in toy parity workflow still clears the `5e-6` bar
- `transport_results_raw.mode == "full_finite"` remains present
- a top-level `runtime_cert` remains present
- `progress.jsonl` still reaches `worker_done`

## Real Continuity Root

Authoritative continuity root:

- `tmp/iter35_kwant_walltime`

I retired the older duplicated-thickness continuity attempt:

- cancelled Kwant job `60185`
- cancelled native-RGF job `60186`

Fresh live local driver after the grouped-thickness relaunch:

```text
PID 412383
/home/msj/Desktop/playground/electroics/wtec/.venv/bin/python .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/iter35_kwant_walltime --queue g4 --walltime 01:00:00
```

Fresh live jobs:

- Kwant: `60193` on `n019`
- native RGF: `60194` on `n020`

The relaunch preserved the continuity root correctly:

- partial Kwant checkpoint was reused
- cached native results through `d07_em0p1` were reused
- the fresh native frontier is now `d07_e0p0`

## New Kwant Runtime Evidence

The grouped-thickness scheduler is now active on the real PBS `qsub` + native `mpirun` path with no fork launchstyle.

Fresh local log evidence:

```text
[wtec][runtime] start 2026-03-13T15:48:41+09:00
[kwant-bench] solver=mumps mumps=True tasks=35 mpi=16 threads=4 length_uc=24
[kwant-bench] resume completed=5/35 pending=30
[kwant-bench] distribution load_min=0 load_max=10985 counts=[5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] first_wave_thicknesses=[3, 5, 7, 9, 11, 13]
[kwant-bench][rank=0] start thickness_uc=3 energy_abs_ev=13.604600
[kwant-bench][rank=1] start thickness_uc=5 energy_abs_ev=13.604600
[kwant-bench][rank=2] start thickness_uc=7 energy_abs_ev=13.604600
[kwant-bench][rank=3] start thickness_uc=9 energy_abs_ev=13.604600
[kwant-bench][rank=4] start thickness_uc=11 energy_abs_ev=13.604600
[kwant-bench][rank=5] start thickness_uc=13 energy_abs_ev=13.604600
```

Current local authority state is still incomplete:

```json
{
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

Current fresh local heartbeat evidence:

```text
[kwant-bench][rank=0] heartbeat thickness_uc=3 energy_abs_ev=13.604600 elapsed_s=120.0
[kwant-bench][rank=1] heartbeat thickness_uc=5 energy_abs_ev=13.604600 elapsed_s=60.0
```

Remote utilization changed materially on the fresh grouped-thickness run. A direct remote probe at about `29s` elapsed showed:

```text
== n019 ==
load average: 13.84, 22.05, 24.50
Mem: 377 total / 3 used / 373 free GiB
16 live python3 -m wtec.transport.kwant_nanowire_benchmark ranks
each rank around 94.6-96.5% CPU, NLWP=8
```

Interpretation:

- the fresh grouped-thickness run no longer reproduces the earlier `~265 GiB` node-memory footprint from duplicated thickness systems
- the first live wave now starts one rank per pending thickness group instead of duplicating thickness `3/5/7` across many ranks
- the local checkpoint is still flat at `5/35`, and no fresh local `done` row or new shard file has landed yet during this developer pass

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
- `sigma_manifest.json`: present

Current live `d07_e0p0` log:

```text
[wtec][runtime] start 2026-03-13T15:48:44+09:00
[wtec][runtime] sigma_extract_start 2026-03-13T15:48:44+09:00
[wtec][sigma] full_finite_principal_start hr_path=TiS_model_b_c_c_canonical_hr.dat length_uc=24 width_uc=13 thickness_uc=7 energy_ev=13.6046 eta_ev=1e-06
[wtec][sigma] full_finite_principal_geometry_ready norb=16 principal_layer_width=6 pad_x=5 nx_effective=34 lead_dim=8736 slice_widths=6,6,6,10,6
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

1. Verify the grouped-thickness scheduler product commit and the current HEAD that records this handoff.
2. Keep using the authoritative continuity root `tmp/iter35_kwant_walltime`; do not reset the evidence trail.
3. Confirm the fresh local Kwant log now shows:
   - `counts=[5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
   - `first_wave_thicknesses=[3, 5, 7, 9, 11, 13]`
   - first live ranks starting one thickness group each, rather than duplicating thickness `3/5/7`
4. Confirm the fresh remote-utilization shape if possible from the live artifacts:
   - no reproduction of the prior `~265 GiB` Kwant node-memory footprint
   - grouped-thickness run resident on `n019`
5. Continue polling for the first fresh grouped-thickness Kwant completion signal:
   - `kwant_reference.json` advancing beyond `5/35`
   - appearance of `kwant_reference.rank1.jsonl`, `rank2`, or `rank3`
   - a fresh `[kwant-bench][rank=...] done ...` line in `model_b/c/kwant/wtec_job.log`
6. Continue polling the native side until `d07_e0p0` returns its local artifact set.
7. Do not claim acceptance until the benchmark-wide authority set exists locally:
   - `model_b/c/rgf/rgf_raw.json`
   - `model_b/c/comparison_raw.json`
   - `model_b/c/comparison_fit.json`
   - `benchmark_summary.json`
8. Keep the SIESTA / ScaLAPACK restart defect separate from benchmark continuity; it still blocks restart acceptance independently.
