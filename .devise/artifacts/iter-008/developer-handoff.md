# Iteration 8 Developer Handoff

## What Changed

- Added a live mid-run failure path for the nanowire benchmark controller in `wtec/cli.py`.
  - `_run_rgf_benchmark_axis(...)` now re-checks existing Kwant/RGF overlap after each completed RGF point and raises `_PartialOverlapFailure` as soon as the current overlap already proves a failed benchmark decision.
  - `_run_kwant_and_rgf_overlap(...)` now converts that condition into a `_KwantOverlapError`, preserves the partial summary and accumulated RGF rows/jobs, and sets the Kwant cancellation event before unwinding.
  - `benchmark_transport(...)` now accepts a live partial-summary fallback, synthesizes a partial Kwant checkpoint when only log-visible overlap exists, and writes `comparison_partial.*` plus top-level `benchmark_summary.json` even when the controller stops because of a live overlap failure before a complete Kwant checkpoint lands.
- Added focused regression coverage in `tests/test_nanowire_benchmark.py`.
  - New test covers propagation of partial-summary metadata through `_run_kwant_and_rgf_overlap(...)`.
  - New command-level test covers a live partial-overlap failure path that writes `benchmark_summary.json` and `comparison_partial.json`.
  - New axis-level test covers stopping RGF-axis submission immediately when fresh overlap already fails tolerance.

## Validation

### Focused Validation For This Patch

- `.venv/bin/pytest -q tests/test_nanowire_benchmark.py tests/test_nanowire_benchmark_progress.py`
  - `35 passed in 0.39s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or load_cached_transport_results_rejects_full_finite_cache_without_exact_sigma"`
  - `2 passed, 11 deselected in 0.26s`

### Declared Dry-Test Contract

- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"`
  - `4 passed, 3 deselected in 4.43s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"`
  - `4 passed, 9 deselected in 0.22s`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"`
  - `5 passed, 36 deselected in 0.17s`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py`
  - `14 passed in 0.18s`

## Live Command-Path Evidence

### Cleanup

- Found a stale local benchmark controller still running:
  - local PID `609499`
- Found stale benchmark jobs still present remotely:
  - `60278` (`kwc_w13_l24`)
  - `60288` (`rgf_prima_nanow`)
- Terminated the stale local controller and cancelled those benchmark jobs before re-validation.

### Fresh Live Rerun On Current Working Tree

- Re-ran the managed benchmark command on the real cluster-backed path:

```bash
.venv/bin/wtec benchmark-transport \
  examples/sio2_tap_sio2_small/run_small.json \
  --output-dir tmp/devise_transport_benchmark \
  --queue g4 \
  --walltime 01:00:00
```

- Current-HEAD controller output:
  - `[benchmark] model=model_b axis=c: existing overlap already proves failure; skipping new PBS submissions`
  - `[benchmark] summary: /home/msj/Desktop/playground/electroics/wtec/tmp/devise_transport_benchmark/benchmark_summary.json`
  - command exited non-zero with:
    - `Transport benchmark failed for target(s): model_b:c:rgf_partial`

### Artifact Outcome

- Fresh artifacts were written at `2026-03-13 19:56:59 +0900`:
  - `tmp/devise_transport_benchmark/benchmark_summary.json`
  - `tmp/devise_transport_benchmark/model_b/c/comparison_partial.json`
  - `tmp/devise_transport_benchmark/model_b/c/comparison_partial.md`
- `benchmark_summary.json` now records:
  - axis status `failed`
  - reason `rgf_partial_validation_failed`
  - `kwant_job.status = "reused_partial_failure"`
  - `kwant_complete = false`
  - `partial_overlap.status = "failed"`
  - `partial_overlap.overlap_points = 5`
- `comparison_partial.json` now records:
  - `status = "failed"`
  - `overlap_points = 5`
  - `max_abs_err = 0.00869376292351376`
  - `max_rel_err = 0.000255698909515118`

### Remote Queue Evidence

- Post-rerun remote `qstat -u msj` showed no new benchmark jobs submitted by the rerun.
- The previously cancelled benchmark jobs remained only in completed (`C`) state:
  - `60278`
  - `60288`
- This is the real-path confirmation that the controller now stops on already-sufficient exact-sigma overlap instead of launching thickness-3 or later work.

## What Remains Unresolved

### Numerical / Physical

- The exact-sigma thickness-1 overlap still fails the benchmark tolerance on the real cluster root.
- Current failed points remain:
  - `E_rel=-0.2`: abs err `0.00869376292351376`
  - `E_rel=-0.1`: abs err `0.005697624704630755`
  - `E_rel=0.0`: abs err `0.005314468726304256`
  - `E_rel=0.1`: abs err `0.0074295696547039825`
  - `E_rel=0.2`: abs err `0.007054917765984214`
- The orchestration bug is closed on the live command path; the remaining benchmark problem is now the actual RGF-vs-Kwant mismatch.

### Restart Path

- The configured restart path is still expected to fail in remote SIESTA setup because ScaLAPACK is still missing.
- I did not change the restart/bootstrap code in this iteration.

## What The Debugger Should Do Next

1. Re-run the managed benchmark flow on current `HEAD` and confirm the live controller now writes `comparison_partial.*` and `benchmark_summary.json` without launching new PBS work once exact-sigma overlap is already sufficient.
2. Treat the controller/orchestration rule from the user-requested task as satisfied on the real command path unless a regression appears.
3. Focus the next verification pass on the remaining physical/numerical issue:
   - the exact-sigma thickness-1 native RGF still misses the Kwant reference by roughly `5e-3` to `9e-3`.
4. Keep the restart-path caveat open separately until the remote ScaLAPACK/SIESTA setup is repaired.
