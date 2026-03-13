# Iteration 6 Developer Handoff

## What changed

- Landed a contract-compatibility filter for the nanowire benchmark evidence path so stale full-finite RGF results without exact-sigma provenance no longer count as current benchmark overlap.
- `scan_partial_rgf_results(...)` now treats a full-finite RGF point as benchmark-compatible only when:
  - a completed `transport_result.json` explicitly records `full_finite_sigma_source = "kwant_exact"` (via runtime cert or result meta), or
  - an in-flight raw/progress point still carries staged `sigma_left_path` and `sigma_right_path` in the payload.
- `TopoSlabWorkflow._load_cached_transport_results(...)` now rejects cached full-finite RGF results when the current config requires internal exact sigma (`_transport_rgf_internal_sigma_mode = "kwant_exact"`) but the cached result does not advertise that sigma source.

## Why this matters

- The large `~14` absolute-error benchmark failure in the live root was coming from stale thickness-1 RGF results produced before the exact-sigma benchmark contract was active.
- The continuity root already showed that exact-sigma reruns collapse that error from `~14` to `~0.005-0.009`, which is still above tolerance but is a completely different regime.
- Without this patch, the controller and partial comparator could keep treating stale native-sigma outputs as authoritative overlap, which blocks the benchmark from recomputing the correct exact-sigma points and confuses the numerical diagnosis.

## New regression coverage

### Benchmark progress / evidence filtering

- `tests/test_nanowire_benchmark_progress.py::test_scan_partial_rgf_results_skips_full_finite_rows_without_exact_sigma_contract`
  - proves stale full-finite results lacking exact-sigma provenance are skipped
- existing progress/result tests were updated to model real benchmark payloads with staged sigma artifacts so the accepted path remains covered

### Cached result reuse

- `tests/test_orchestrator_json.py::test_load_cached_transport_results_rejects_full_finite_cache_without_exact_sigma`
  - proves cached reuse is disabled when the current full-finite run expects exact sigma but the cached result lacks `full_finite_sigma_source = "kwant_exact"`

## Live evidence rechecked this turn

### Live current benchmark root after contract filtering

- Comparator rerun:
  - `.venv/bin/python -m wtec.transport.nanowire_benchmark_progress --kwant-dir tmp/devise_transport_benchmark/model_b/c/kwant --rgf-root tmp/devise_transport_benchmark/model_b/c/rgf`
- Result now:
  - `status = no_overlap`
  - `overlap_points = 0`
- Important detail:
  - the old thickness-1 current-root rows are now skipped because they do not carry exact-sigma provenance
  - `skipped_payloads` includes the stale `d01_*` and early `d03_*` payloads from the old root
- Interpretation:
  - the previous `~14` error is no longer treated as valid current-contract evidence
  - the live root now needs fresh exact-sigma thickness-1 RGF points before it can make a benchmark decision

### Continuity root still shows the real exact-sigma numerical gap

- Comparator rerun:
  - `.venv/bin/python -m wtec.transport.nanowire_benchmark_progress --kwant-dir tmp/iter35_kwant_walltime/model_b/c/kwant --rgf-root tmp/iter35_kwant_walltime/model_b/c/rgf`
- Result:
  - `status = failed`
  - `overlap_points = 5`
  - `max_abs_err = 0.00869376292410351`
  - `max_rel_err = 0.0002556989095324511`
- Interpretation:
  - the exact-sigma benchmark mismatch is real, but it is the smaller `~5e-3` to `~9e-3` regime seen on the continuity root, not the stale `~14` mismatch from the old current root

## Tests run

### Focused benchmark/orchestrator coverage

- `.venv/bin/pytest -q tests/test_nanowire_benchmark.py tests/test_nanowire_benchmark_progress.py`
  - `32 passed in 0.34s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "stage_transport_rgf_qsub_reuses_cached_result_for_requested_label or load_cached_transport_results_rejects_full_finite_cache_without_exact_sigma or full_finite"`
  - `3 passed, 10 deselected in 0.24s`

### Declared dry-test contract

- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"`
  - `4 passed, 3 deselected in 4.43s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"`
  - `4 passed, 9 deselected in 0.22s`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"`
  - `5 passed, 36 deselected in 0.17s`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py`
  - `14 passed in 0.18s`

## Files changed

- `wtec/transport/nanowire_benchmark_progress.py`
- `wtec/workflow/orchestrator.py`
- `tests/test_nanowire_benchmark_progress.py`
- `tests/test_orchestrator_json.py`

## What remains unresolved

- I did not re-run the real PBS-backed `benchmark-transport` command after this patch, so the debugger still needs to confirm that the live current root now launches fresh exact-sigma thickness-1 RGF work instead of reusing the stale native-sigma results.
- The exact-sigma continuity root still fails the raw tolerance with `max_abs_err = 0.00869376292410351`, so the actual physical/numerical benchmark mismatch is not solved by this iteration.
- The clean restart path still fails in remote SIESTA/ScaLAPACK setup; this iteration did not touch it.
- The `>=5x` speedup acceptance is still unresolved because correctness/tolerance has not yet cleared on valid exact-sigma benchmark evidence.

## What the debugger should do next

1. Re-run the real managed benchmark command on `tmp/devise_transport_benchmark` and confirm the controller no longer accepts or reuses the stale thickness-1 native-sigma cache.
2. Verify that fresh exact-sigma thickness-1 RGF work is launched when needed and that new result/runtime-cert files record `full_finite_sigma_source = "kwant_exact"`.
3. Once valid exact-sigma thickness-1 overlap exists on the live root, re-check the benchmark comparator and determine whether the remaining mismatch matches the continuity-root `~0.005-0.009` regime.
4. Keep the orchestration fix from iteration 4/5 intact: when valid exact-sigma overlap is already sufficient to fail, the controller should still emit `comparison_partial.*` and `benchmark_summary.json` without launching unnecessary new work.
