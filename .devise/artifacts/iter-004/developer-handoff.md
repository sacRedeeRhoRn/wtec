# Iteration 4 Developer Handoff

## What changed

- Patched `benchmark-transport` so it inspects existing Kwant/RGF overlap before any new PBS submission when the full Kwant checkpoint is still incomplete.
- If the existing overlap already fails tolerance, the controller now:
  - writes `comparison_partial.json`
  - writes `comparison_partial.md`
  - records an axis-level failed summary with `reason = "rgf_partial_validation_failed"`
  - skips launching fresh Kwant / native-RGF jobs for that axis
- Extended `_write_partial_nanowire_axis_artifacts(...)` to accept a precomputed summary so the controller can reuse the same overlap decision it made before launch.
- Hardened the partial-progress scanner so unfinished RGF progress payloads without a completed `native_point_done` do not abort overlap comparison; they are skipped until they become usable evidence.

## Why this addresses the debugger finding

- The iteration-3 report showed the controller was still resuming `kwant_reference.json` and then submitting new PBS work even though the existing root already contained failed overlap visible from `wtec_job.log` plus completed RGF transport artifacts.
- The landed controller change moves that decision point ahead of new job submission and uses the same partial-overlap machinery that can see Kwant log rows, not just `kwant_reference.json`.
- The scanner hardening removes another live-root failure mode where a still-running RGF point could block comparison of already-completed overlap.

## Added regression coverage

### New CLI decision-path regression

- `tests/test_nanowire_benchmark.py::test_benchmark_transport_uses_existing_log_overlap_before_launch`
- Coverage:
  - existing incomplete `kwant_reference.json`
  - Kwant evidence available only in `wtec_job.log`
  - existing completed RGF point already violates tolerance
  - fresh `submit_kwant_nanowire_reference(...)` and `_run_rgf_benchmark_axis(...)` are both monkeypatched to fail if called
- Verified outcome:
  - command exits failed
  - `comparison_partial.json` exists
  - `benchmark_summary.json` exists
  - the partial evidence was sourced from `kwant_log`
  - no new submit path was taken

### New partial-progress resilience regression

- `tests/test_nanowire_benchmark_progress.py::test_compare_partial_benchmark_progress_skips_incomplete_rgf_progress`
- Coverage:
  - one completed overlap point
  - one extra in-flight RGF payload with only `worker_start`
- Verified outcome:
  - comparator still returns the completed failed overlap
  - the unfinished payload is skipped instead of crashing the comparison

## Evidence rechecked this turn

### Live current benchmark root

- Comparator rerun:
  - `.venv/bin/python -m wtec.transport.nanowire_benchmark_progress --kwant-dir tmp/devise_transport_benchmark/model_b/c/kwant --rgf-root tmp/devise_transport_benchmark/model_b/c/rgf`
- Current result:
  - `status = failed`
  - `overlap_points = 5`
  - `max_abs_err = 14.005380636980433`
  - `max_rel_err = 0.3183041053859334`
- Interpretation:
  - the live root already contains enough completed overlap to justify an immediate failed decision without launching new PBS work

### Continuity root

- Comparator rerun:
  - `.venv/bin/python -m wtec.transport.nanowire_benchmark_progress --kwant-dir tmp/iter35_kwant_walltime/model_b/c/kwant --rgf-root tmp/iter35_kwant_walltime/model_b/c/rgf`
- Current result:
  - `status = failed`
  - `overlap_points = 5`
  - `max_abs_err = 0.00869376292410351`
  - `max_rel_err = 0.0002556989095324511`
- Interpretation:
  - the continuity root still shows the same five-point benchmark-tolerance failure

## Tests run

### Focused benchmark coverage

- `.venv/bin/pytest -q tests/test_nanowire_benchmark.py tests/test_nanowire_benchmark_progress.py`
  - `31 passed in 0.26s`

### Declared dry-test contract

- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"`
  - `4 passed, 3 deselected in 4.98s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"`
  - `3 passed, 9 deselected in 0.30s`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"`
  - `5 passed, 36 deselected in 0.27s`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py`
  - `14 passed in 0.29s`

## Files changed

- `wtec/cli.py`
- `wtec/transport/nanowire_benchmark_progress.py`
- `tests/test_nanowire_benchmark.py`
- `tests/test_nanowire_benchmark_progress.py`

## What remains unresolved

- I did not rerun the full real `wtec benchmark-transport ...` command against the live cluster root in this turn, so the debugger still needs to confirm that the command now exits early and emits the summary artifacts on an actual PBS-backed use flow.
- Remote job cancellation after an already-proven failed overlap is still not implemented.
- The clean restart path still has the previously reported remote ScaLAPACK/SIESTA blocker; this iteration did not touch it.
- The `>=5x` speedup acceptance remains unproven because the benchmark evidence is still failing on the currently completed overlap.

## What the debugger should do next

1. Re-run the real managed benchmark command on the existing partial root.
2. Confirm that when failed overlap already exists, the command now:
   - does not launch fresh Kwant / native-RGF PBS jobs
   - writes `model_b/c/comparison_partial.json`
   - writes `model_b/c/comparison_partial.md`
   - writes top-level `benchmark_summary.json`
   - exits deterministically with the failed benchmark decision
3. Confirm that the live partial comparator still works while additional RGF points remain in flight.
4. Separately keep the restart-path ScaLAPACK/SIESTA failure and any job-cancellation follow-up as unresolved verification items.
