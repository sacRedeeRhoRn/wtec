# Iteration 12 Developer Handoff

## What Changed

- Fixed the real benchmark CLI crash in [`wtec/cli.py`](wtec/cli.py):
  - `benchmark_transport(...)` now imports `resolve_transport_rgf_eta` from `wtec.workflow.orchestrator` in the scope where it is actually used.
- Added a command-path regression in [`tests/test_nanowire_benchmark_cluster.py`](tests/test_nanowire_benchmark_cluster.py):
  - the new test executes `benchmark_transport.callback(...)` through the exact-sigma eta-resolution branch on a partial-overlap root
  - it verifies the branch is reachable without the previous `NameError`
  - it also verifies the command still produces `benchmark_summary.json` and `comparison_partial.json` from existing partial overlap in that path

## Why This Iteration Was Needed

- Iteration 10 introduced the exact-sigma eta contract but the real managed use flow still crashed before it could launch fresh `1e-8` work.
- The debugger isolated the failure to:
  - `NameError: name 'resolve_transport_rgf_eta' is not defined`
  - raised inside `benchmark_transport(...)`
- The symbol existed in `wtec/workflow/orchestrator.py`, but the `benchmark_transport(...)` scope in `wtec/cli.py` did not import it.

## Declared Dry-Test Contract

- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"`
  - `4 passed, 3 deselected in 4.75s`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"`
  - `5 passed, 10 deselected in 0.17s`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"`
  - `5 passed, 36 deselected in 0.14s`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py`
  - `15 passed in 0.22s`

## Focused Regression

- `pytest -q tests/test_nanowire_benchmark_cluster.py::test_benchmark_transport_reaches_exact_sigma_eta_branch_without_name_error`
  - `1 passed in 0.23s`

## Real Cluster-Backed Verification

I re-ran the actual benchmark command on the live managed root:

```bash
.venv/bin/wtec benchmark-transport \
  examples/sio2_tap_sio2_small/run_small.json \
  --output-dir tmp/devise_transport_benchmark \
  --queue g4 \
  --walltime 01:00:00
```

### Old blocker cleared

- The command got past the old crash point.
- Observed live output:
  - resumed the partial Kwant reference under `tmp/devise_transport_benchmark/model_b/c/kwant/kwant_reference.json`
  - printed `launching Kwant and native RGF in parallel`
  - submitted fresh remote jobs instead of failing in Python before launch

### Fresh exact-sigma `1e-8` runtime evidence

Fresh local payloads/results were created on the managed benchmark root for thickness `1`:

- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p2/transport/primary/transport_result.json`
  - `eta = 1e-08`
  - `G = 33.99991303555754`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_em0p1/transport/primary/transport_result.json`
  - `eta = 1e-08`
  - `G = 37.99994301687156`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_e0p0/transport/primary/transport_result.json`
  - `eta = 1e-08`
  - `G = 39.99994685099053`
- `tmp/devise_transport_benchmark/model_b/c/rgf/d01_e0p1/transport/primary/transport_result.json`
  - `eta = 1e-08`
  - `G = 43.99992567886085`

Remote/local runtime evidence remained on the required path:

- fresh payloads use:
  - `transport_rgf_mode = "full_finite"`
  - `eta = 1e-08`
  - `sigma_left_path` / `sigma_right_path`
- fresh runtime certs report:
  - `queue = "g4"`
  - `mpi_size = 1`
  - `omp_threads = 64`
  - `full_finite_sigma_source = "kwant_exact"`

Observed live remote jobs during verification:

- Kwant:
  - `60305`
- fresh exact-sigma RGF jobs:
  - `60306`, `60310`, `60311`, `60312`

### Fresh partial-overlap status after four new exact-sigma points

After the fresh `1e-8` thickness-1 points were written, I re-ran:

```python
compare_partial_benchmark_progress(
    kwant_dir="tmp/devise_transport_benchmark/model_b/c/kwant",
    rgf_root="tmp/devise_transport_benchmark/model_b/c/rgf",
    required_exact_eta=1.0e-8,
)
```

Current result:

- `status = "ok"`
- `overlap_points = 4`
- `checked_points = 4`
- `max_abs_err = 8.696444147204829e-05`
- `max_rel_err = 2.557777690354436e-06`

Interpretation:

- The real exact-sigma mismatch that previously sat in the `~5e-3` to `~9e-3` range is now effectively gone on the four freshly recomputed thickness-1 overlap points.
- This is direct live-cluster evidence that the `1e-8` eta contract is doing the intended physical/numerical work on the real benchmark path.

## Cleanup Performed

- I terminated the local verification driver process after collecting enough evidence to avoid leaving another stale benchmark controller behind.
- I also cancelled the fresh Kwant verification job `60305` after the needed RGF-side evidence had been captured.
- Final remote scheduler state after cleanup:
  - `60305`, `60306`, `60310`, `60311`, `60312` all ended in completed state (`C`) in `qstat`

## Remaining Unresolved

- The configured restart/bootstrap path is still broken:
  - `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python`
  - remote SIESTA setup still fails because ScaLAPACK is missing
- I stopped the verification run intentionally before waiting for the final thickness-1 `+0.2 eV` point or later benchmark stages to fully settle under the current root.
- The project still does **not** have a new final benchmark summary proving:
  - full raw/fit acceptance on the live root
  - `>=5x` speedup versus the current Kwant baseline

## What The Debugger Should Do Next

1. Resume the real benchmark flow from the current `tmp/devise_transport_benchmark` root on this commit.
2. Verify that the remaining fresh exact-sigma thickness-1 point(s) are also generated under `eta = 1e-8`.
3. Re-run the partial comparator after the remaining overlap points arrive and confirm the overlap still stays inside tolerance.
4. Once thickness-1 is fully green, continue to the next benchmark layer instead of revisiting the now-fixed CLI crash.
5. Keep the restart-path ScaLAPACK/SIESTA issue tracked separately; this iteration does not change that blocker.
