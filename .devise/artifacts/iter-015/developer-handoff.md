## Iteration 15 Developer Handoff

### Outcome

- Status: `needs_verifier`
- Commit intent: tighten real-benchmark full-finite parity instead of only making the native-RGF path observable.
- Goal status: materially improved on the real cluster path, but still not closed in this turn.

### What Changed

1. Patched the full-finite left-lead coupling in `wtec/ext/rgf/src/rgf_runner.c`.
   - The full-finite solver now builds a raw left coupling block, reverses the lead rows with `wtec_reverse_lead_rows(...)`, and only then forms the left self-energy.
   - This aligns the full-finite left-lead embedding with the existing periodic solver, which was already reversing the left interface orientation.

2. Removed the extra transverse-span x padding from the full-finite benchmark path in `wtec/ext/rgf/src/rgf_runner.c`.
   - `pad_x` now uses only `p_eff - 1`.
   - The old `+ max_ry + max_rz` inflation was silently extending the effective x length beyond the requested benchmark `length_uc=24`.
   - The now-unused `wtec_active_transverse_spans_full(...)` helper was deleted.

3. Bumped the native runner binary id to force a fresh remote scaffold rebuild.
   - `wtec/rgf.py`: `RGF_BINARY_ID = "wtec_rgf_runner_phase2_v6"`
   - `wtec/ext/rgf/include/wtec_rgf.h`: `WTEC_RGF_BINARY_ID "wtec_rgf_runner_phase2_v6"`

4. Tightened benchmark router reuse in `wtec/cli.py`.
   - `_ensure_nanowire_benchmark_rgf_router_ready(...)` now reuses init-state only when the recorded `binary_id` matches the current `RGF_BINARY_ID`.
   - This prevents the benchmark helper from silently reusing a stale remote scaffold after C-side solver changes.

5. Added focused coverage in `tests/test_nanowire_benchmark.py`.
   - Existing router-ready tests now assert against `RGF_BINARY_ID`.
   - New test: `test_ensure_nanowire_benchmark_rgf_router_ready_rebuilds_stale_binary`

### Why This Change Was Needed

- Iteration 14 proved the benchmark was no longer blocked on staging: it was already submitting real native-RGF transport jobs on the cluster.
- The remaining blocker was now a real parity failure:
  - Kwant thickness-1 benchmark points were in the `34-44 G0` range.
  - Native full-finite RGF was still returning `15-24 G0` on the same first points.
- Two concrete asymmetries stood out in the solver:
  1. the periodic path reversed the left interface block but the full-finite path did not
  2. the full-finite path was inflating effective x length with extra `max_ry + max_rz` padding that the benchmark contract did not request

### Fresh Evidence From This Turn

#### 1. The real benchmark improved across three successive native-runner builds

Same benchmark point, same benchmark workspace, same cluster route:

| runner | point | G_mean |
| --- | --- | --- |
| `v4` | `d01_em0p2` (`E=13.4046 eV`) | `15.07822487744949` |
| `v5` | `d01_em0p2` (`E=13.4046 eV`) | `24.53102940394606` |
| `v6` | `d01_em0p2` (`E=13.4046 eV`) | `26.60817684230457` |

Interpretation:

- The left-lead coupling patch materially improved the first real benchmark point.
- The x-padding reduction improved it again and also reduced the effective slice count in the full-finite solve.
- The benchmark mismatch is smaller than before, but still far outside any acceptable tolerance.

#### 2. The exact benchmark command is live on the updated `v6` scaffold

Current local benchmark driver:

- PID `4193635`

Current init-state router record:

```json
{
  "ready": true,
  "binary_id": "wtec_rgf_runner_phase2_v6",
  "numerical_status": "phase2_experimental"
}
```

What this clears:

- The benchmark helper is no longer reusing an old remote binary after native-solver edits.
- The live benchmark is exercising the `v6` runner, not the earlier `v4`/`v5` builds.

#### 3. The live benchmark is still on the real native-RGF cluster path

Examples of current `v6` benchmark transport artifacts:

- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p2/transport/primary/transport_result.json`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p2/transport/primary/transport_runtime_cert.json`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d03_em0p2/transport/primary/transport_result.json`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d03_em0p2/transport/primary/transport_runtime_cert.json`

Representative runtime-cert evidence from the current run:

```json
{
  "queue": "g4",
  "mode": "full_finite",
  "mpi_size": 1,
  "omp_threads": 64,
  "wall_seconds": 7.142633,
  "effective_thread_count": 48.11011
}
```

This means the benchmark is still executing through:

- remote PBS `qsub`
- native `mpirun`
- real cluster-side `full_finite` transport jobs

and the no-fork launchstyle requirement remains intact.

#### 4. The benchmark trace shows the overlap branch is healthy

Current trace file:

- `tmp/devise_transport_benchmark/model_b/c/rgf_launch_trace.jsonl`

Trace summary at handoff time:

```json
{
  "started": 7,
  "done": 6,
  "exceptions": 0
}
```

Latest trace milestones:

```json
[
  {"event":"rgf_case_done","tag":"d03_em0p2","job_id":"60000"},
  {"event":"rgf_case_start","tag":"d03_em0p1"},
  {"event":"rgf_case_before_from_config","tag":"d03_em0p1"},
  {"event":"rgf_case_before_stage_transport","tag":"d03_em0p1"}
]
```

Interpretation:

- The earlier staging blind spot is gone.
- The live run has advanced beyond thickness-1 and is now staging the next thickness-3 case.

#### 5. The real benchmark still fails parity on the first completed overlapping points

Kwant reference values below are the live thickness-1 completions already captured by the debugger from the same active benchmark run. Native-RGF values are from the current `v6` local transport artifacts:

| energy (eV) | Kwant | RGF `v6` | abs delta |
| --- | --- | --- | --- |
| `13.4046` | `34.000000000001` | `26.60817684230457` | `7.391823157696432` |
| `13.5046` | `37.999999999997` | `28.35033079172224` | `9.649669208274762` |
| `13.6046` | `40.0` | `29.88663916904678` | `10.11336083095322` |
| `13.7046` | `43.999999999999` | `29.99461936301757` | `14.005380636981428` |

Interpretation:

- This is still a real numerical/physical mismatch, not just a missing-artifact problem.
- The benchmark now needs another solver-level parity fix before the final comparison files can possibly pass.

#### 6. The exact-sigma diagnostic probe was attempted but not resolved

- I started a local exact-sigma extraction probe, but it was too slow without a better sparse backend and I terminated it.
- I also submitted a cluster-side exact-sigma probe job:
  - job id `59998`
- That probe did not produce usable evidence quickly enough, so I cancelled it explicitly:
  - `qdel 59998`

### Declared Dry Tests

All configured dry-test shards passed after the code changes:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
2 passed, 3 deselected in 2.61s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
2 passed, 8 deselected in 0.19s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.21s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
6 passed in 0.20s
```

Focused benchmark coverage also passed:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "ensure_nanowire_benchmark_rgf_router_ready or append_nanowire_benchmark_trace or build_nanowire_benchmark_source_seed or resolve_nanowire_benchmark_source_structure or run_kwant_and_rgf_overlap or select_benchmark_models_defaults_to_primary_rgf_model or build_tis_benchmark_source_cfg_uses_explicit_source_nodes or tis_"
11 passed, 6 deselected in 0.26s
```

### Still Unresolved

1. The configured restart command is still not green.
   - `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python`
   - still fails during remote SIESTA preparation because ScaLAPACK is missing.

2. The real benchmark still has not cleared parity.
   - `kwant_reference.json` is not written yet.
   - `comparison_raw.json`, `comparison_fit.json`, and `benchmark_summary.json` are not written yet.
   - the required `>= 5x` speedup remains unproven.

3. The current bottleneck is now solver behavior, not cluster staging.
   - The first `v6` overlapping points are still systematically below the live Kwant reference.

### What The Debugger Should Do Next

1. Reuse the already-running `v6` local benchmark driver instead of launching a duplicate run.
   - Local PID: `4193635`

2. Continue from the active benchmark workspace:
   - `tmp/devise_transport_benchmark/model_b/c/rgf_launch_trace.jsonl`
   - `tmp/devise_transport_benchmark/model_b/c/rgf/*/transport/primary/transport_result.json`
   - `tmp/devise_transport_benchmark/model_b/c/rgf/*/transport/primary/transport_runtime_cert.json`

3. Treat the next issue as a real parity investigation, not a routing investigation.
   - The benchmark now stages and completes real native-RGF transport cases on the cluster.
   - The remaining mismatch is in the conductance values themselves.

4. Compare the current `v6` benchmark cases against Kwant as the run advances.
   - Thickness-1 already shows the benchmark is closer than `v4`, but still wrong.
   - Thickness-3 is now entering the same path and should reveal whether the mismatch is systematic across thickness.

5. If another solver fix is attempted next, keep the router binary-id freshness guard.
   - It prevented reuse of stale remote scaffolds and made the `v6` validation credible.

6. Keep the restart-path gap separate from the benchmark parity issue.
   - `wtec init` still needs its own ScaLAPACK/SIESTA resolution.
   - The benchmark parity problem is now independently reproducible on the working cluster transport path.
