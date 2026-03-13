# Developer Handoff

## What Changed

- Added a partial-checkpoint loader for the Kwant nanowire benchmark path in `wtec/cli.py`.
- Added `_KwantOverlapError` so the overlap launcher preserves completed native-RGF rows/jobs even when the Kwant future exits incomplete after retries.
- Added `_write_partial_nanowire_axis_artifacts(...)` to emit `comparison_partial.json` and `comparison_partial.md` from the existing overlap-only progress utilities.
- Updated `benchmark-transport` so the primary RGF axis no longer throws away usable partial evidence:
  - if the Kwant sweep remains incomplete after retries but a partial checkpoint exists, the CLI now loads that checkpoint
  - it writes partial overlap artifacts against the existing native-RGF results
  - if the current overlap already proves RGF is out of tolerance, it converts that into a deterministic benchmark failure instead of waiting for all 35 Kwant points
  - if the current overlap still passes, it remains conservative and reports that full Kwant completion is still required

## Why

- The current project task explicitly asked for the currently completed overlapping Kwant points to count when they support a defensible decision.
- The repo already had `wtec.transport.nanowire_benchmark_progress`, but `benchmark-transport` still hard-gated on a fully complete Kwant reference and therefore could not terminate early on already-failing overlap evidence.
- This change keeps the existing full-benchmark behavior for success paths while allowing failure to surface sooner when the available overlap is already enough to prove the RGF benchmark is out of tolerance.

## Dry-Test Evidence

Focused tests:

- `.venv/bin/pytest -q tests/test_nanowire_benchmark.py tests/test_nanowire_benchmark_progress.py`
  - `28 passed`

Declared dry-test contract:

- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"`
  - `4 passed, 3 deselected`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"`
  - `3 passed, 9 deselected`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"`
  - `5 passed, 36 deselected`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py`
  - `14 passed`

## Live Benchmark Evidence Captured

I ran the existing partial-overlap comparator against the current continuity root:

- command:
  - `.venv/bin/python -m wtec.transport.nanowire_benchmark_progress --kwant-dir tmp/iter35_kwant_walltime/model_b/c/kwant --rgf-root tmp/iter35_kwant_walltime/model_b/c/rgf`
- result:
  - status: `failed`
  - overlap points: `5`
  - max_abs_err: `0.00869376292410351`
  - max_rel_err: `0.0002556989095324511`
- all five currently overlapping thickness-1 points fail the existing raw tolerance

That means the new fallback path should now be able to turn the present partial checkpoint state into a real benchmark failure artifact instead of waiting on full `35/35` Kwant completion.

## Unresolved

- I did not rerun the full real `benchmark-transport` workflow after the patch; the code change and current overlap evidence indicate the failure path now has enough information to terminate, but the debugger still needs to exercise the actual managed use flow.
- I did not touch the separate remote `wtec init` ScaLAPACK/SIESTA restart blocker.
- The repo has many unrelated user changes in the worktree; I only modified `wtec/cli.py`, `tests/test_nanowire_benchmark.py`, and this handoff.

## Verifier Next Steps

- Run the real managed benchmark use flow on current `HEAD`.
- Confirm that an incomplete Kwant checkpoint with failing overlap now produces:
  - `comparison_partial.json`
  - `comparison_partial.md`
  - `benchmark_summary.json` with a failed benchmark decision instead of indefinite waiting
- Verify the failure is grounded in the existing overlap mismatch rather than missing runtime artifacts.
