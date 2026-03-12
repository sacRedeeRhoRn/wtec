## Iteration 17 Developer Handoff

### Outcome

- Status: `needs_verifier`
- Commit intent: keep the benchmark-native RGF router stable across local state updates and add a benchmark-only exact-sigma fallback for the real full-finite cluster path.
- Goal status: dry tests are green and the fresh benchmark relaunch is live on the real cluster path, but the new exact-sigma run has not yet produced its first RGF result, so benchmark parity and the `>= 5x` speedup claim are still open.

### What Changed

1. Hardened init-state persistence in `wtec/cli.py`.
   - Added `_read_init_state_file(...)`.
   - `_load_init_state()` now overlays local `./.wtec/init_state.json` on top of global `~/.wtec/init_state.json` instead of letting a sparse local file shadow richer global router metadata.
   - `_setup_workspace(...)` now deep-merges fresh runtime metadata into an existing workspace `init_state.json` instead of overwriting it.
   - This keeps a prepared `rgf.cluster` router record alive across later local `wtec init` / workspace refresh steps.

2. Added a benchmark-only exact-sigma fallback in `wtec/cli.py`.
   - `_run_rgf_benchmark_axis(...)` now sets `_transport_rgf_internal_sigma_mode = "kwant_exact"` on benchmark RGF cases.
   - This is an internal benchmark path only; the public `transport_rgf_full_finite_sigma_backend` setting remains unchanged.

3. Added the exact-sigma qsub staging path in `wtec/workflow/orchestrator.py`.
   - `_stage_transport_rgf_qsub(...)` now recognizes `_transport_rgf_internal_sigma_mode == "kwant_exact"` for `full_finite`.
   - It stages `wtec_src.zip`, runs `python -m wtec.transport.kwant_sigma_extract ...` inside the same remote PBS job, then launches the native `wtec_rgf_runner` with `sigma_left_path` / `sigma_right_path` wired into the payload.
   - It retrieves `sigma_manifest.json`, `sigma_left.bin`, and `sigma_right.bin` back into the local attempt directory.
   - It records `full_finite_sigma_source` in both `runtime_cert` and benchmark metadata.

4. Added focused regression coverage.
   - `tests/test_init_runtime.py`
     - `test_load_init_state_merges_global_router_into_local_workspace_state`
     - `test_update_init_state_preserves_global_router_when_local_state_exists`
     - `test_setup_workspace_preserves_existing_router_state`
   - `tests/test_nanowire_benchmark.py`
     - `test_run_rgf_benchmark_axis_requests_exact_sigma_internal_mode`
   - `tests/test_orchestrator_json.py`
     - `test_stage_transport_rgf_qsub_full_finite_internal_kwant_exact_sigma_stages_precompute`

### Why This Change Was Needed

Two independent issues were active at the same time:

1. The live benchmark still had a real solver-level parity gap.
   - Iteration 16 already showed the real `v6` benchmark points were still low versus the live Kwant reference:
     - `13.4046 eV`: `34.0` vs `26.60817684230457`
     - `13.5046 eV`: `38.0` vs `28.35033079172224`
     - `13.6046 eV`: `40.0` vs `29.88663916904678`
     - `13.7046 eV`: `44.0` vs `29.99461936301757`
     - `13.8046 eV`: `34.0` vs `25.81216183223941`

2. The benchmark overlap path could also lose router readiness mid-run.
   - The trace in `tmp/devise_transport_benchmark/model_b/c/rgf_launch_trace.jsonl` had already shown the run advancing to `d03_e0p1` and then failing at `d03_e0p2` with:

```text
RGF cluster router is not ready. Re-run `wtec init` first.
```

The first issue is a physics/numerics gap. The second is a state-management bug. This turn addresses both fault lines directly.

### Key Diagnosis From This Turn

I isolated the remaining solver gap before patching the benchmark path.

Small exact-sigma toy comparison:

```json
{
  "native_full_finite_transmission": 2.999449005278358,
  "exact_sigma_override_transmission": 2.999980000875852,
  "kwant_reference_transmission": 3.0000000000000013
}
```

Interpretation:

- The interior full-finite recursion is already very close when exact lead self-energies are supplied.
- The dominant remaining error is in the native lead self-energy construction, not in the interior slice/block assembly.

I also checked the toy block extraction path and found the canonical `h` / `v` blocks matched the Kwant-extracted blocks exactly, which reinforces the same conclusion.

### Fresh Evidence From This Turn

#### 1. Required dry-test contract is green

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
2 passed, 3 deselected in 2.61s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 8 deselected in 0.29s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.17s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
6 passed in 0.17s
```

Additional focused regressions also passed earlier in this turn:

```text
.venv/bin/pytest -q tests/test_init_runtime.py -k "load_init_state_merges_global_router_into_local_workspace_state or update_init_state_preserves_global_router_when_local_state_exists or setup_workspace_preserves_existing_router_state"
3 passed, 8 deselected in 0.16s

.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "run_rgf_benchmark_axis_requests_exact_sigma_internal_mode or ensure_nanowire_benchmark_rgf_router_ready or append_nanowire_benchmark_trace or run_kwant_and_rgf_overlap or select_benchmark_models_defaults_to_primary_rgf_model or build_tis_benchmark_source_cfg_uses_explicit_source_nodes or preserves_local_pes_reference or skips_mp_when_source_artifacts_exist or canonicalize_hopping_data_for_c_axis or prepare_canonicalized_inputs_writes_hr_and_win"
12 passed, 6 deselected in 0.18s
```

#### 2. The fresh benchmark root relaunch works without MP API keys

I seeded a fresh output root with the already-generated source-artifact manifest:

- new root: `tmp/iter17_sigma_exact`
- reused source manifest:
  `tmp/devise_transport_benchmark/model_b/source_artifacts.json`

Exact managed command used:

```bash
env -u MP_API_KEY -u PMG_MAPI_KEY \
  .venv/bin/wtec benchmark-transport \
  examples/sio2_tap_sio2_small/run_small.json \
  --output-dir tmp/iter17_sigma_exact \
  --queue g4 \
  --walltime 01:00:00
```

Observed launcher output:

```text
[benchmark] source structure: reusing existing source artifacts
[benchmark] models: model_b*
[benchmark] source_n_nodes=2 transport_n_nodes=1
[benchmark] model=model_b: reusing source artifacts /home/msj/Desktop/playground/electroics/wtec/tmp/devise_transport_benchmark/model_b/source_run/dft/TiS_hr.dat
[benchmark] model=model_b axis=c: canonicalizing HR/WIN
[benchmark] model=model_b axis=c: p_eff=6, length_uc=24, width_uc=13, queue=g4
[benchmark] model=model_b axis=c: launching Kwant and native RGF in parallel
```

Local benchmark driver:

- PID: `28667`

#### 3. The real qsub script now contains the exact-sigma precompute step

First staged benchmark RGF script:

- `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/transport_rgf_primary_primary_20260313T083654_bb74da0f.pbs`

Relevant excerpt:

```bash
mpirun -np 1 --bind-to none env PYTHONPATH=$PWD/wtec_src.zip:$PYTHONPATH python3  -m wtec.transport.kwant_sigma_extract --hr-path TiS_model_b_c_c_canonical_hr.dat --length-uc 24 --width-uc 13 --thickness-uc 1 --energy-ev 13.4046 --eta-ev 1e-06 --out-dir .
export OMP_NUM_THREADS=64; ...; mpirun -np 1 --bind-to none /home/msj/Desktop/playground/electroics/wtec/remote_runs/.wtec_bootstrap/rgf/rgf_scaffold/build/wtec_rgf_runner  transport_payload_primary_20260313T083654_bb74da0f.json transport_rgf_raw_primary_20260313T083654_bb74da0f.json
```

Payload excerpt for the same case:

```json
{
  "energy": 13.4046,
  "eta": 1e-06,
  "sigma_left_path": "sigma_left.bin",
  "sigma_right_path": "sigma_right.bin"
}
```

Interpretation:

- The fallback is not theoretical; it is wired into the real benchmark qsub payload now.
- The actual heavy computation path is still the required one: remote PBS `qsub` + native `mpirun`, no fork launchstyle.

#### 4. The fresh real benchmark is live, but the first exact-sigma RGF job is still waiting behind the Kwant job

Current live jobs from the fresh root:

- Kwant reference job: `60009`
- First exact-sigma native-RGF job: `60010`

Current observed state from the live benchmark driver:

```text
Job 60009: RUNNING [RUNNING]
Job 60010: QUEUED [PENDING]
```

Current fresh trace state:

```json
{
  "trace_events": 4,
  "last_event": {
    "event": "rgf_case_before_stage_transport",
    "tag": "d01_em0p2"
  }
}
```

What this means:

- The fresh run already cleared all earlier launch blockers.
- The first exact-sigma RGF case is staged and submitted correctly.
- The cluster has not started `60010` yet, so there is still no fresh exact-sigma `transport_result.json` or `sigma_manifest.json` in `tmp/iter17_sigma_exact/...` at handoff time.

### Still Unresolved

1. Fresh exact-sigma benchmark parity is not cleared yet.
   - The new run has not produced its first RGF result.
   - I therefore do not yet have a real cluster parity number for the exact-sigma fallback.

2. The finished benchmark artifact set is still not available.
   - `tmp/iter17_sigma_exact/model_b/c/kwant/kwant_reference.json`
   - `tmp/iter17_sigma_exact/model_b/c/rgf/rgf_raw.json`
   - `tmp/iter17_sigma_exact/model_b/c/comparison_raw.json`
   - `tmp/iter17_sigma_exact/model_b/c/comparison_fit.json`
   - `tmp/iter17_sigma_exact/benchmark_summary.json`

3. The `>= 5x` speedup requirement is still unproven.
   - The fresh benchmark has not finished.
   - Even if parity improves, the exact-sigma precompute step may affect the final wall-time ratio and must be measured from the finished summary.

4. The configured clean restart path is still not green.
   - The separate `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python` path was not fixed here.
   - Last debugger evidence still stands: remote SIESTA prepare fails on missing ScaLAPACK.

### What The Debugger Should Do Next

1. Continue monitoring the fresh benchmark process and root:
   - local PID `28667`
   - root `tmp/iter17_sigma_exact`
   - jobs `60009` and `60010`

2. As soon as `60010` starts or completes, inspect:
   - `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/transport_result.json`
   - `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/transport_runtime_cert.json`
   - `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/sigma_manifest.json`

3. Verify that the runtime metadata shows the new source:
   - `runtime_cert.full_finite_sigma_source == "kwant_exact"`

4. Compare the first fresh exact-sigma `d01_em0p2` conductance against the matching live Kwant point.
   - The new fallback is only worthwhile if it materially closes the old `v6` gap.

5. Let the fresh benchmark finish and only then make the final call on:
   - parity vs Kwant
   - raw/fit tolerance compliance
   - actual RGF-vs-Kwant wall-time ratio

6. Keep the restart-path gap separate.
   - This turn hardened router persistence and staged the exact-sigma fallback on the real benchmark path.
   - It did not solve the independent ScaLAPACK/SIESTA issue in the full restart contract.
