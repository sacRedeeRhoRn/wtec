# Iteration 11 Developer Handoff

## Verdict

- Status: `needs_fix`
- Branch: `devise/wtec-rgf-completion/developer`
- Declared dry tests: green
- Exact benchmark relaunch without `MP_API_KEY`: fixed
- Goal fully met: no

This iteration removes the immediate launch blocker that stopped the exact managed benchmark command in a shell without `MP_API_KEY`. The benchmark driver now reuses existing source artifacts before demanding MP-backed PES resolution, and it also preserves a caller-provided local `dft_pes_reference_structure_file` when MP resolution is actually needed. The exact benchmark command now advances past the old API-key usage error and reaches the overlap branch again. The deeper overlap-stage blocker is still open: after the relaunch clears MP validation, the command still does not emit the first native-RGF benchmark run directory or transport artifact.

## What Changed

### 1. Added a reusable nanowire benchmark source-seed helper

File:

- `wtec/cli.py`

New helper:

- `_build_nanowire_benchmark_source_seed(...)`

Purpose:

- centralizes the benchmark source-resolution seed fields
- preserves caller-provided values from the benchmark config
- specifically threads through:
  - `dft_pes_reference_structure_file`
  - `dft_pes_reference_mp_id`
  - `dft_pes_reference_use_primitive`
  - `mp_api_key`
  - `mp_api_key_env`

This fixes the earlier bug where `benchmark-transport` rebuilt a reduced seed and silently dropped the caller-supplied local PES reference path before `_ensure_pes_reference_structure_from_mp(...)` ran.

### 2. Added reuse-aware benchmark source-structure resolution

File:

- `wtec/cli.py`

New helper:

- `_resolve_nanowire_benchmark_source_structure(...)`

Behavior:

- checks `model_root/source_artifacts.json` for each selected benchmark model before resolving the source structure
- returns an empty string when all selected models already have reusable source artifacts
- only calls `_ensure_pes_reference_structure_from_mp(...)` when some selected model still needs source generation

Why:

- the previous implementation validated Materials Project access before it checked whether the current workspace already had the needed source artifacts
- that prevented the exact benchmark command from relaunching from the existing `tmp/devise_transport_benchmark/model_b/source_artifacts.json` state

### 3. Updated `benchmark-transport` to honor the new source-resolution ordering

File:

- `wtec/cli.py`

Behavior change:

- the command now prints:

```text
[benchmark] source structure: reusing existing source artifacts
```

when the selected benchmark models already have source artifacts

- if a model still needs source generation, the command resolves the source structure only at that point, using the full caller-aware seed instead of the reduced hardcoded seed

Result:

- the exact configured benchmark command no longer fails immediately at:

```text
Error: dft_pes_reference_mp_id is set but Materials Project API key is missing.
```

when the current benchmark workspace already contains reusable source artifacts

### 4. Added focused unit coverage for the fixed source-resolution contract

File:

- `tests/test_nanowire_benchmark.py`

New tests:

- `test_build_nanowire_benchmark_source_seed_preserves_local_pes_reference`
- `test_resolve_nanowire_benchmark_source_structure_skips_mp_when_source_artifacts_exist`

Coverage intent:

- proves the local PES reference survives seed construction
- proves benchmark source reuse now bypasses MP-backed structure resolution entirely when source artifacts already exist

## Test Evidence

Focused benchmark tests:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "run_kwant_and_rgf_overlap or build_nanowire_benchmark_source_seed or resolve_nanowire_benchmark_source_structure or select_benchmark_models_defaults_to_primary_rgf_model or build_tis_benchmark_source_cfg_uses_explicit_source_nodes or tis_"
.......                                                                  [100%]
7 passed, 6 deselected in 0.22s
```

Declared dry-test contract:

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
..                                                                       [100%]
2 passed, 3 deselected in 2.60s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
..                                                                       [100%]
2 passed, 8 deselected in 0.15s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
.....                                                                    [100%]
5 passed, 36 deselected in 0.18s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
......                                                                   [100%]
6 passed in 0.18s
```

## Real Benchmark Evidence

### Exact managed command before this fix

The debugger report showed the exact command failed immediately without `MP_API_KEY`:

```text
env -u MP_API_KEY -u PMG_MAPI_KEY .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/devise_transport_benchmark --queue g4 --walltime 01:00:00
```

with:

```text
Error: dft_pes_reference_mp_id is set but Materials Project API key is missing. Set MP_API_KEY (or PMG_MAPI_KEY), or provide dft_pes_reference_structure_file.
```

### Exact managed command after this fix

I re-ran the same command with `MP_API_KEY` and `PMG_MAPI_KEY` explicitly unset:

```text
env -u MP_API_KEY -u PMG_MAPI_KEY .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/devise_transport_benchmark --queue g4 --walltime 01:00:00
```

Observed output:

```text
[benchmark] source structure: reusing existing source artifacts
[benchmark] models: model_b*
[benchmark] source_n_nodes=2 transport_n_nodes=1
[benchmark] model=model_b: reusing source artifacts /home/msj/Desktop/playground/electroics/wtec/tmp/devise_transport_benchmark/model_b/source_run/dft/TiS_hr.dat
[benchmark] model=model_b axis=c: canonicalizing HR/WIN
[benchmark] model=model_b axis=c: p_eff=6, length_uc=24, width_uc=13, queue=g4
[benchmark] model=model_b axis=c: launching Kwant and native RGF in parallel
```

Interpretation:

- the launch-path bug is fixed
- the exact managed command now clears the old MP-key validation barrier
- the current benchmark workspace is now correctly reused without needing MP access

### Remote evidence from the fixed relaunch

The fixed exact relaunch did submit real cluster work again:

- local driver PID: `4158968`
- remote job: `59963`

Remote queue snapshot during the relaunch:

```text
59963  msj  g4  kwc_w13_l24  R
```

This confirms the command once again gets far enough to use the real runtime path:

- remote PBS `qsub`
- native `mpirun`
- no fork launchstyle

### Remaining live blocker

Even after clearing the MP launch failure, the benchmark still did not advance into the first native-RGF benchmark run directory:

Observed local tree remained limited to:

```text
tmp/devise_transport_benchmark/model_b/c
tmp/devise_transport_benchmark/model_b/c/canonical
tmp/devise_transport_benchmark/model_b/c/kwant
tmp/devise_transport_benchmark/model_b/c/kwant_overlap_quiet_snapshot_20260313T071900
tmp/devise_transport_benchmark/model_b/c/kwant_overlap_snapshot_20260313T071341
tmp/devise_transport_benchmark/model_b/c/kwant_single_rank_snapshot_20260313T065522
```

Still absent:

- no `tmp/devise_transport_benchmark/model_b/c/rgf/...` directory
- no benchmark `kwant_reference.json`
- no benchmark `transport_result.json`
- no benchmark `transport_runtime_cert.json`
- no `benchmark_summary.json`

I cancelled the relaunch after capturing the fixed launch behavior so it would not keep consuming cluster time while still parked before the first native-RGF benchmark artifact.

## What This Iteration Cleared

- The exact managed benchmark command can now relaunch from the current workspace without `MP_API_KEY`.
- Caller-provided local PES references are now preserved in the benchmark source seed.
- Source-artifact reuse now happens before any MP-backed structure fetch is attempted.
- The declared full-finite regression gates remain green.

## What Remains Unresolved

- The overlap-stage blocker is still present after the launch-path fix.
- The benchmark still does not create the first `model_b/c/rgf/...` run directory.
- The benchmark still has no `kwant_reference.json`.
- The benchmark still has no native-RGF `transport_result.json`.
- The benchmark still has no native-RGF `transport_runtime_cert.json`.
- The benchmark still has no `benchmark_summary.json`.
- The `>= 5x` speedup requirement remains unproven.
- `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python` is still non-green because remote SIESTA preparation still fails on missing ScaLAPACK.

## What The Debugger Should Do Next

1. Re-run the exact managed benchmark command from the current workspace without `MP_API_KEY` and verify it reproduces the now-fixed launch behavior instead of the old immediate usage error.

2. Continue from the overlap-stage blocker only after confirming the launch-path fix:

   - the command reaches `launching Kwant and native RGF in parallel`
   - a real Kwant job is submitted
   - but no `model_b/c/rgf/...` directory appears

3. Debug the overlap branch from that later state, focusing on:

   - `_run_kwant_and_rgf_overlap(...)` in `wtec/cli.py`
   - the call into `_run_rgf_benchmark_axis(...)`
   - why the first native-RGF benchmark transport directory is never materialized

4. Only after the benchmark writes:

   - `kwant_reference.json`
   - `transport_result.json`
   - `transport_runtime_cert.json`
   - `benchmark_summary.json`

   should any parity or `>= 5x` speedup claim be revisited.
