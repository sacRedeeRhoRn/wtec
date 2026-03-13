# Iteration 10 Developer Handoff

## What Changed

- Added an explicit exact-sigma eta resolver in [`wtec/workflow/orchestrator.py`](wtec/workflow/orchestrator.py):
  - default RGF eta remains `1e-6`
  - full-finite internal `kwant_exact` runs now default to `1e-8` unless `transport_rgf_eta` is explicitly set
- Tightened cached full-finite exact-sigma reuse:
  - cached `transport_result.json` is now rejected when `full_finite_sigma_source != kwant_exact`
  - cached exact-sigma results are also rejected when `transport_results_raw.eta` does not match the active exact-sigma eta contract
- Threaded the exact-sigma eta contract through the nanowire benchmark CLI:
  - `_run_rgf_benchmark_axis(...)` now accepts `required_exact_eta`
  - the benchmark command now computes the exact-sigma eta once per primary axis and injects it into each launched RGF point via `transport_rgf_eta`
  - partial-overlap artifact generation and prelaunch/live overlap checks now use the same required eta
- Tightened live partial-progress scanning in [`wtec/transport/nanowire_benchmark_progress.py`](wtec/transport/nanowire_benchmark_progress.py):
  - when `required_exact_eta` is provided, full-finite exact-sigma rows are skipped unless both payload/result eta match that contract
  - this prevents stale `1e-6` exact-sigma rows from counting as valid overlap after the benchmark switches to `1e-8`

## Why This Patch

- The remaining thickness-1 exact-sigma mismatch was consistent with over-broadening rather than the earlier controller bug.
- Earlier in this turn I ran a direct remote PBS `qsub + mpirun` probe for the current `d01_em0p2` exact-sigma benchmark point with `eta=1e-8`.
- That probe returned native RGF transmission `33.99991303555754` against the existing Kwant reference `33.99999999999901`, shrinking the pointwise absolute error to about `8.7e-05`.
- The current benchmark root had been generated with exact-sigma `eta=1e-6`, which is large enough to explain the old thickness-1 error band (`~5e-3` to `~9e-3`).

## Tests And Evidence

### Focused extra tests

- `.venv/bin/pytest -q tests/test_nanowire_benchmark_progress.py tests/test_nanowire_benchmark.py::test_run_rgf_benchmark_axis_requests_exact_sigma_internal_mode tests/test_orchestrator_json.py::test_resolve_transport_rgf_eta_defaults_to_small_eta_for_exact_sigma tests/test_orchestrator_json.py::test_load_cached_transport_results_rejects_full_finite_cache_with_exact_sigma_eta_mismatch tests/test_orchestrator_json.py::test_stage_transport_rgf_qsub_full_finite_internal_kwant_exact_sigma_stages_precompute`
  - `12 passed in 0.28s`

### Declared dry-test contract

- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"`
  - `4 passed, 3 deselected in 21.41s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"`
  - `5 passed, 10 deselected in 0.63s`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"`
  - `5 passed, 36 deselected in 0.30s`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py`
  - `14 passed in 0.30s`

### Live root check after the patch

- Re-ran the partial comparator against the current benchmark root with the new exact-sigma eta contract:
  - `compare_partial_benchmark_progress(kwant_dir="tmp/devise_transport_benchmark/model_b/c/kwant", rgf_root="tmp/devise_transport_benchmark/model_b/c/rgf", required_exact_eta=1e-8)`
- Result:
  - `status = "missing_rgf_points"`
  - `overlap_points = 0`
  - `required_exact_eta = 1e-8`
  - `skipped_payloads = 22`
- Interpretation:
  - stale exact-sigma `1e-6` rows no longer count as reusable/valid overlap
  - the current root now needs fresh exact-sigma `1e-8` RGF results before any new partial comparison is meaningful

## Files Changed

- `wtec/workflow/orchestrator.py`
- `wtec/transport/nanowire_benchmark_progress.py`
- `wtec/cli.py`
- `tests/test_orchestrator_json.py`
- `tests/test_nanowire_benchmark_progress.py`
- `tests/test_nanowire_benchmark.py`

## Remaining Unresolved

- I did not run a full fresh benchmark-transport campaign to completion on the live cluster root after this patch.
- The debugger still needs to verify that a fresh exact-sigma `1e-8` benchmark rerun closes the thickness-1 mismatch on the real command path and then re-evaluate fit/speed evidence.
- The configured restart/bootstrap path is still unresolved:
  - `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python`
  - remote SIESTA setup still fails when CMake cannot find ScaLAPACK

## What The Debugger Should Do Next

1. Re-run the managed real benchmark command on `tmp/devise_transport_benchmark` with current `HEAD`.
2. Confirm the command does **not** reuse old exact-sigma `1e-6` transport results.
3. Confirm fresh RGF payloads/runtime certs for the new run carry exact-sigma `eta=1e-8`.
4. Re-check the five thickness-1 overlap points first; if the probe behavior holds, the prior `~0.005` to `~0.009` error band should collapse.
5. If thickness-1 clears tolerance, continue to the next benchmark mismatch layer instead of reopening the controller/orchestration path.
6. Re-run or re-check the restart path separately; this patch does not address the remote ScaLAPACK/SIESTA blocker.
