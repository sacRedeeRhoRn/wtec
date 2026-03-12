# Developer Handoff

## What Changed

- Split the benchmark node budget so the QE/Wannier source build can use a separate `--source-nodes` setting while the actual Kwant-vs-RGF comparison keeps the transport-stage node count from the run config.
- `wtec benchmark-transport` now defaults `--source-nodes` to `2`, records `source_n_nodes` and `transport_n_nodes` in the benchmark summary metadata, and prints both values at launch.
- `_build_tis_benchmark_source_cfg()` now stamps the explicit source-build node count into the benchmark source workflow config.
- The Kwant reference submission and native-RGF transport submissions now both explicitly use the transport-stage node budget, so the source-build acceleration does not silently change the fairness envelope of the benchmark comparison.
- Added `test_build_tis_benchmark_source_cfg_uses_explicit_source_nodes()` to lock the new source-node contract.
- Cancelled the older one-node source-build benchmark (`PID 4081474`, remote job `59934`), archived its partial workspace to `tmp/devise_transport_benchmark_one_node_snapshot_20260313T063224`, and relaunched the managed benchmark command on the patched path.

## What Passed

- `.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "select_benchmark_models_defaults_to_primary_rgf_model or build_tis_benchmark_source_cfg_uses_explicit_source_nodes or tis_"` -> `4 passed, 6 deselected`
- `.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py tests/test_pbs_resources.py` -> `8 passed`
- `.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"` -> `2 passed, 3 deselected`
- `.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"` -> `2 passed, 8 deselected`
- `.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"` -> `5 passed, 36 deselected`
- `.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py` -> `5 passed`

## Real Benchmark Evidence

- Relaunched the managed benchmark command with the patched default source-node split:
  - `.venv/bin/wtec benchmark-transport examples/sio2_tap_sio2_small/run_small.json --output-dir tmp/devise_transport_benchmark --queue g4 --walltime 01:00:00`
- The relaunched live process is:
  - `PID 4107024`
  - it now prints `[benchmark] source_n_nodes=2 transport_n_nodes=1`
- Source-build queue evidence on the patched path:
  - `59943` `scf_TiS` completed on `2` nodes / `128` MPI ranks
  - `59944` `nscf_TiS` completed on `2` nodes / `128` MPI ranks
  - `59945` `w90_TiS` completed on `2` nodes / `128` MPI ranks
- The patched `wannier90` PBS script is now:
  - `select=2:ncpus=64:mpiprocs=64:ompthreads=1`
  - `mpirun -np 128 --bind-to core pw2wannier90.x ...`
  - `mpirun -np 128 --bind-to core wannier90.x TiS`
- The source build now reaches a clean Wannier completion instead of stalling upstream of transport:
  - `Time to disentangle bands     12.871 (sec)`
  - `Time for wannierise           55.564 (sec)`
  - `Total Execution Time         108.437 (sec)`
  - `All done: wannier90 exiting`
- Local source artifacts now exist for the real benchmark path:
  - `tmp/devise_transport_benchmark/model_b/source_artifacts.json`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS_hr.dat`
  - `tmp/devise_transport_benchmark/model_b/source_run/dft/TiS.wout`
  - `tmp/devise_transport_benchmark/model_b/c/canonical/TiS_model_b_c_c_canonical_hr.dat`
- The live benchmark has advanced beyond source generation and is now in the real Kwant reference stage:
  - remote job `59946`
  - remote worker log: `[kwant-bench] solver=mumps mumps=True tasks=35 mpi=1 threads=64 length_uc=24`
  - first task started: `thickness_uc=1 energy_abs_ev=13.404600`

## Remaining Unresolved

- The active benchmark has not finished the Kwant baseline yet, so native-RGF transport has not started in this iteration and there is still no benchmark `transport_result.json`, `transport_runtime_cert.json`, or `benchmark_summary.json`.
- The required `>= 5x` wall-time speedup versus the Kwant baseline remains unproven until the live benchmark completes the Kwant stage and then runs the native-RGF stage on the same reference case.
- The configured restart command still fails during remote SIESTA preparation because ScaLAPACK is missing on the cluster. That remains separate from the QE/Wannier benchmark route.

## What The Debugger Should Do Next

- Continue monitoring the active live benchmark process `PID 4107024` and remote Kwant job `59946` rooted at `tmp/devise_transport_benchmark`.
- Once `59946` completes, verify that the benchmark immediately advances into the native-RGF qsub path and produces `transport_result.json`, `transport_runtime_cert.json`, and transport progress artifacts under the benchmark tree.
- After the benchmark writes `benchmark_summary.json`, quantify the real Kwant-vs-RGF wall-time ratio and decide whether the `>= 5x` speedup acceptance bar is met.
- Keep the current source-node split unless real transport evidence disproves it: it cleared the benchmark’s previous upstream source bottleneck and reached the Kwant reference stage in the same workspace.
