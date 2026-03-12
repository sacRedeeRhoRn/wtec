## Iteration 13 Developer Handoff

### Outcome

- Status: `needs_verifier`
- Commit intent: benchmark now self-heals the native-RGF router instead of dying at the init-state gate.
- Goal status: improved materially, not fully closed in this turn.

### What Changed

1. Added benchmark-scoped native-RGF router preparation in `wtec/cli.py`.
   - New helper `_ensure_nanowire_benchmark_rgf_router_ready(...)` checks init-state for an acceptable cluster router record.
   - When the router is missing, the benchmark now runs `_prepare_cluster_rgf_router_setup(dry_run=False)`, persists the returned cluster capability into init-state, and refuses to continue only if the scaffold still fails to become ready.

2. Added narrow benchmark tracing in `wtec/cli.py`.
   - New helper `_append_nanowire_benchmark_trace(...)` writes JSONL events.
   - `_run_rgf_benchmark_axis(...)` now records per-case milestones around `TopoSlabWorkflow.from_config(...)` and `_stage_transport_rgf_qsub(...)`.
   - The trace also creates the first `axis_dir/rgf/<tag>` run directory immediately, which makes the first native-RGF staging point observable.

3. Added focused tests in `tests/test_nanowire_benchmark.py`.
   - `test_append_nanowire_benchmark_trace_writes_jsonl`
   - `test_ensure_nanowire_benchmark_rgf_router_ready_reuses_ready_state`
   - `test_ensure_nanowire_benchmark_rgf_router_ready_prepares_missing_state`

### Why This Change Was Needed

- The previous live benchmark already reached the overlap branch but stalled before any native-RGF benchmark artifacts appeared.
- The new trace captured the actual first exception:

```text
RuntimeError: RGF cluster router is not ready. Re-run `wtec init` first.
```

- Both `.wtec/init_state.json` and `~/.wtec/init_state.json` lacked any `rgf.cluster` record because the configured `wtec init` path still aborts earlier in remote SIESTA preparation when ScaLAPACK is missing.
- The benchmark therefore needed a targeted self-heal for the native-RGF cluster scaffold instead of relying on the full restart path.

### Fresh Evidence From This Turn

#### 1. The new self-heal path executed successfully

Exact benchmark command:

```bash
env -u MP_API_KEY -u PMG_MAPI_KEY \
  .venv/bin/wtec benchmark-transport \
  examples/sio2_tap_sio2_small/run_small.json \
  --output-dir tmp/devise_transport_benchmark \
  --queue g4 \
  --walltime 01:00:00
```

Observed launcher output:

```text
[benchmark] native RGF router: preparing cluster scaffold
✓ native RGF scaffold ready: /home/msj/Desktop/playground/electroics/wtec/remote_runs/.wtec_bootstrap/rgf/rgf_scaffold/build/wtec_rgf_runner
```

Result:

- `.wtec/init_state.json` now contains a ready `rgf.cluster` record.
- Recorded router binary id: `wtec_rgf_runner_phase2_v4`
- Recorded numerical status: `phase2_experimental`

#### 2. The real benchmark now stages native-RGF transport jobs

Current local benchmark driver:

- PID `4174451`

Current live benchmark trace:

- `tmp/devise_transport_benchmark/model_b/c/rgf_launch_trace.jsonl`
- At handoff time:
  - `started = 6`
  - `done = 5`
  - `exceptions = 0`
  - latest case entered `d03_em0p2`

Examples from the trace:

```json
{"event":"rgf_case_done","job_id":"59974","tag":"d01_em0p2"}
{"event":"rgf_case_done","job_id":"59975","tag":"d01_em0p1"}
{"event":"rgf_case_done","job_id":"59976","tag":"d01_e0p0"}
```

#### 3. Real native-RGF benchmark artifacts now exist locally

Examples already written by the live run:

- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p2/transport/primary/transport_result.json`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p2/transport/primary/transport_runtime_cert.json`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p1/transport/primary/transport_result.json`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_e0p0/transport/primary/transport_result.json`

Live runtime-cert evidence from the first completed benchmark case:

```json
{
  "engine": "rgf",
  "binary_id": "wtec_rgf_runner_phase2_v4",
  "numerical_status": "phase2_experimental",
  "mode": "full_finite",
  "queue": "g4",
  "mpi_size": 1,
  "omp_threads": 64,
  "wall_seconds": 7.655311
}
```

#### 4. Real cluster-path evidence remains intact

Active remote jobs at handoff included:

- Kwant: `59973` (`kwc_w13_l24`)
- Native RGF: `59979` (`rgf_prima_nanow`)

This means the exact benchmark path is now simultaneously exercising:

- remote PBS `qsub`
- native `mpirun`
- Kwant reference stage
- native-RGF transport stage

and no fork launchstyle was introduced by this fix.

### Declared Dry Tests

All configured dry-test shards passed after the code changes:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
2 passed, 3 deselected in 2.58s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
2 passed, 8 deselected in 0.14s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.15s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
6 passed in 0.18s
```

Focused new coverage also passed:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "ensure_nanowire_benchmark_rgf_router_ready or append_nanowire_benchmark_trace or build_nanowire_benchmark_source_seed or resolve_nanowire_benchmark_source_structure or run_kwant_and_rgf_overlap or select_benchmark_models_defaults_to_primary_rgf_model or build_tis_benchmark_source_cfg_uses_explicit_source_nodes or tis_"
10 passed, 6 deselected in 0.22s
```

### Still Unresolved

1. The configured restart command is still not green.
   - `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python`
   - still fails during remote SIESTA preparation because ScaLAPACK is missing.

2. The live benchmark has not finished the full 35-point Kwant-vs-RGF comparison yet.
   - `kwant_reference.json` not yet written at handoff time.
   - `benchmark_summary.json` not yet written at handoff time.
   - full raw/fit comparison status and the required `>= 5x` speedup are therefore still unproven in this turn.

### What The Debugger Should Do Next

1. Reuse the already-running local benchmark driver instead of launching a duplicate run.
   - Local PID: `4174451`

2. Monitor the active benchmark until it either completes or exposes the next concrete failure.
   - Primary trace file:
     - `tmp/devise_transport_benchmark/model_b/c/rgf_launch_trace.jsonl`
   - Existing benchmark transport artifacts:
     - `tmp/devise_transport_benchmark/model_b/c/rgf/*/transport/primary/transport_result.json`
     - `tmp/devise_transport_benchmark/model_b/c/rgf/*/transport/primary/transport_runtime_cert.json`

3. Once the live run finishes, verify the final artifact set:
   - `tmp/devise_transport_benchmark/model_b/c/kwant/kwant_reference.json`
   - `tmp/devise_transport_benchmark/model_b/c/rgf_raw.json`
   - `tmp/devise_transport_benchmark/model_b/c/comparison_raw.json`
   - `tmp/devise_transport_benchmark/model_b/c/comparison_fit.json`
   - `tmp/devise_transport_benchmark/benchmark_summary.json`

4. Use `benchmark_summary.json` to make the final call on:
   - raw/fit parity against the Kwant reference path
   - actual wall-time speedup versus the Kwant baseline

5. Keep the restart-path gap separate from the benchmark-router fix.
   - The benchmark no longer needs to wait on the full SIESTA path to stage native-RGF transport.
   - The ScaLAPACK/SIESTA failure still remains an operational issue for the configured `wtec init` contract.
