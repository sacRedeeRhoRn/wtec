# Developer Handoff

## What Changed

- Added `select_benchmark_models()` in `wtec.transport.nanowire_benchmark` so `benchmark-transport` runs only the primary RGF-bearing model by default and still supports an explicit all-models path.
- Added `--all-models/--primary-model-only` to `wtec benchmark-transport`, echoed the selected models, and recorded `model_scope` plus `selected_model_keys` in the benchmark summary metadata.
- Added `test_select_benchmark_models_defaults_to_primary_rgf_model()` to lock the new selection contract.
- Stopped the earlier model-a-first benchmark attempt, preserved its local workspace snapshot under `tmp/devise_transport_benchmark_model_a_snapshot_*`, cancelled remote job `59927`, and relaunched the managed benchmark command so the real cluster run proceeds directly on `model_b`.

## What Passed

- `.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "select_benchmark_models_defaults_to_primary_rgf_model or tis_"` -> `3 passed, 6 deselected`
- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"` -> `2 passed, 3 deselected`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"` -> `2 passed, 8 deselected`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"` -> `5 passed, 36 deselected`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py tests/test_nanowire_benchmark.py -k "not progress"` -> `14 passed`

## Real Benchmark Evidence

- Relaunched the managed benchmark command with a shell-provided `MP_API_KEY` after preserving the earlier dotenv fix:
  - `MP_API_KEY=... .venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/devise_transport_benchmark --queue g4 --walltime 01:00:00`
- The new run now prints `[benchmark] models: model_b*`, confirming the primary-only selector is active for the managed benchmark path.
- Current real-cluster PBS evidence for the relaunched `model_b` benchmark:
  - `59932` `scf_TiS` -> `COMPLETED` in `00:01:12`
  - `59933` `nscf_TiS` -> `COMPLETED` in `00:02:33`
  - `59934` `w90_TiS` -> `RUNNING` when this handoff was written
- The active `wannier90` output is converging rather than failing immediately on the old `dis_windows` error; the live log progressed past iteration `166` on the disentanglement loop for `model_b`.
- The local benchmark tree is now the primary-model workspace:
  - `tmp/devise_transport_benchmark/references/TiS_primitive_mp-1018028.cif`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS.scf.out`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS.nscf.out`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS.pw2wan.in`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS.win`

## Remaining Unresolved

- The primary-model benchmark has not reached the Kwant-reference or native-RGF transport stages yet in this iteration, so there is still no `benchmark_summary.json`, no `kwant_reference.json`, no native-RGF `transport_result.json`, and no `transport_runtime_cert.json`.
- The required `>= 5x` wall-time speedup versus Kwant remains unproven until the active `model_b` benchmark finishes and writes the summary artifacts.
- `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python` is still known to fail in this workspace during remote SIESTA preparation because ScaLAPACK is missing on the cluster. That remains separate from the QE/Wannier benchmark route.

## What The Debugger Should Do Next

- Resume monitoring the live benchmark session rooted at `tmp/devise_transport_benchmark` and confirm that the active `model_b` run completes past `w90_TiS` into the Kwant-reference and native-RGF transport stages.
- Once `model_b` finishes, verify the benchmark writes `benchmark_summary.json`, `kwant_reference.json`, native-RGF `transport_result.json`, and `transport_runtime_cert.json`, and confirm those transport artifacts were produced via remote PBS `qsub` + native `mpirun` without fork launchstyle.
- Quantify the RGF-vs-Kwant wall-time ratio from the resulting summary and decide whether the primary-model-only default is sufficient for the managed acceptance path or whether a later article-completeness pass should rerun with `--all-models`.
