## Iteration 19 Developer Handoff

### Outcome

- Status: `needs_verifier`
- Commit intent: prevent failed overlap runs from leaking stale Kwant benchmark jobs that block the queue, while keeping the exact-sigma benchmark relaunch alive on the real cluster path.
- Goal status: the declared dry-test contract is green, the stale-queue blockage is fixed in code and cleared in the live workspace, and the fresh exact-sigma RGF job is now actually running. Fresh exact-sigma parity, final benchmark summary, `>= 5x` speedup proof, and the separate ScaLAPACK/SIESTA restart-path failure remain open.

### What Changed

1. Patched overlap cancellation cleanup in `wtec/cli.py`.
   - `_run_kwant_and_rgf_overlap(...)` now creates a cancellation event for the concurrent Kwant branch.
   - If `run_rgf_axis()` raises, the overlap helper sets that event, waits for the Kwant future to observe cancellation, and then re-raises the original RGF exception.
   - The benchmark CLI now passes the cancellation event through to `submit_kwant_nanowire_reference(...)`.

2. Patched `wtec/transport/nanowire_benchmark_cluster.py`.
   - `submit_kwant_nanowire_reference(...)` now accepts `cancel_event`.
   - It forwards that event into `JobManager.submit_and_wait(...)`.
   - If the wait path ends with a qdel-triggered failure while cancellation is active, it translates that into `CancelledError` so the overlap helper can cleanly suppress the obsolete Kwant side instead of surfacing a second noisy failure.

3. Patched `wtec/cluster/submit.py`.
   - `JobManager.wait(...)` and `submit_and_wait(...)` now accept `cancel_event`.
   - While polling, if `cancel_event.is_set()` and the job is still non-terminal, `wait(...)` now issues `qdel <job_id>` once and continues polling until the scheduler reports a terminal state.
   - This gives the overlap helper a real way to stop a no-longer-useful remote PBS job instead of merely abandoning a local future.

4. Added focused regression coverage.
   - `tests/test_nanowire_benchmark.py`
     - updated `test_run_kwant_and_rgf_overlap_runs_rgf_while_kwant_waits`
     - added `test_run_kwant_and_rgf_overlap_cancels_kwant_when_rgf_raises`
   - `tests/test_nanowire_benchmark_cluster.py`
     - added `test_submit_kwant_nanowire_reference_forwards_cancel_event`

### Why This Change Was Needed

The fresh exact-sigma benchmark from iteration 17/18 was initially not moving for a reason that was **not** a new solver failure:

- fresh benchmark root: `tmp/iter17_sigma_exact`
- fresh exact-sigma RGF job: `60010`
- fresh Kwant job: `60009`

The real queue blocker was an **older failed benchmark process** that was still alive:

- stale local PID: `4193635`
- stale output root: `tmp/devise_transport_benchmark`
- stale trace had already ended with:

```text
RGF cluster router is not ready. Re-run `wtec init` first.
```

Even after that overlap branch had failed, its old Kwant job `59990` was still running on `g4`, holding one of the four `g4` slots and keeping the fresh exact-sigma job `60010` queued.

Root cause in code:

- `_run_kwant_and_rgf_overlap(...)` launched the Kwant branch in a background future.
- If the RGF branch later raised, nothing cancelled the already-running Kwant job.
- The future just kept running until its remote PBS job finished, leaking queue occupancy.

This turn fixes that failure mode directly.

### Fresh Evidence From This Turn

#### 1. Required dry-test contract is green

```text
.venv/bin/pytest -q tests/test_rgf_native_runner.py -k "full_finite or disorder_ensembles"
2 passed, 3 deselected in 2.56s

.venv/bin/pytest -q tests/test_orchestrator_json.py -k "full_finite or rgf_qsub_canonicalizes_axes or routes_rgf_to_native_qsub"
3 passed, 8 deselected in 0.19s

.venv/bin/pytest -q tests/test_preflight_config.py -k "full_finite or phase2"
5 passed, 36 deselected in 0.17s

.venv/bin/pytest -q tests/test_kwant_block_extract.py tests/test_rgf_postprocess.py tests/test_nanowire_benchmark_cluster.py
7 passed in 0.18s
```

Focused cleanup regressions also passed:

```text
.venv/bin/pytest -q tests/test_nanowire_benchmark.py -k "run_kwant_and_rgf_overlap"
2 passed, 17 deselected in 0.19s

.venv/bin/pytest -q tests/test_nanowire_benchmark_cluster.py -k "conservative_multi_rank_layout or forwards_cancel_event"
2 passed, 1 deselected in 0.16s
```

#### 2. I confirmed the stale queue blockage and cleared it in the live workspace

Live local benchmark processes before cleanup:

```text
PID 28667 -> .venv/bin/wtec benchmark-transport ... --output-dir tmp/iter17_sigma_exact
PID 4193635 -> .venv/bin/wtec benchmark-transport ... --output-dir tmp/devise_transport_benchmark
```

The old trace for `tmp/devise_transport_benchmark` already showed the run had failed at `d03_e0p2`:

```json
{
  "event": "rgf_case_exception",
  "tag": "d03_e0p2",
  "exc_type": "RuntimeError",
  "exc_message": "RGF cluster router is not ready. Re-run `wtec init` first."
}
```

But the old Kwant job was still occupying `g4`:

```text
59990 kwc_w13_l24 R g4
60009 kwc_w13_l24 R g4
60010 rgf_prima_nanow Q g4
```

I manually reclaimed that obsolete slot:

- `qdel 59990`
- `kill 4193635`

Result:

```text
59990 kwc_w13_l24 C g4
60009 kwc_w13_l24 R g4
60010 rgf_prima_nanow R g4
```

This was a live-workspace cleanup step, not the code fix itself. The code fix above is what prevents future overlap failures from leaking the same stale queue occupancy.

#### 3. The fresh exact-sigma benchmark is still the real cluster path

Fresh root:

- `tmp/iter17_sigma_exact`

Fresh live local driver:

- PID `28667`

Fresh live queue state at handoff:

- Kwant job `60009` -> `RUNNING`
- exact-sigma RGF job `60010` -> `RUNNING`

The first staged fresh benchmark script remains the same exact-sigma route verified in iteration 17/18:

```bash
mpirun -np 1 --bind-to none env PYTHONPATH=$PWD/wtec_src.zip:$PYTHONPATH python3 -m wtec.transport.kwant_sigma_extract --hr-path TiS_model_b_c_c_canonical_hr.dat --length-uc 24 --width-uc 13 --thickness-uc 1 --energy-ev 13.4046 --eta-ev 1e-06 --out-dir .
...
mpirun -np 1 --bind-to none /home/msj/Desktop/playground/electroics/wtec/remote_runs/.wtec_bootstrap/rgf/rgf_scaffold/build/wtec_rgf_runner transport_payload_primary_20260313T083654_bb74da0f.json transport_rgf_raw_primary_20260313T083654_bb74da0f.json
```

Interpretation:

- The fresh benchmark is still exercising real PBS `qsub` + native `mpirun`.
- No fork launchstyle is involved.
- The old queue stall was not a routing bug in the fresh exact-sigma job; it was stale occupancy from the earlier failed overlap run.

#### 4. The fresh exact-sigma job is running, but the first measured result is still not local yet

Current local artifact state under `tmp/iter17_sigma_exact`:

```json
{
  "benchmark_summary": false,
  "kwant_reference": false,
  "result_count": 0,
  "cert_count": 0,
  "sigma_manifest_count": 0
}
```

Current fresh trace still stops before local retrieval of the first case:

```json
[
  {"event":"rgf_axis_start"},
  {"event":"rgf_case_start","tag":"d01_em0p2"},
  {"event":"rgf_case_before_from_config","tag":"d01_em0p2"},
  {"event":"rgf_case_before_stage_transport","tag":"d01_em0p2"}
]
```

Remote live status for the same case:

- remote dir:
  `/home/msj/Desktop/playground/electroics/wtec/remote_runs/nanowire_rgf_model_b_c_d01_em0p2/transport/primary`
- `wtec_job_primary_20260313T083654_bb74da0f.log` currently contains only:

```text
[wtec][runtime] start 2026-03-13T08:53:24+09:00
```

Interpretation:

- The fresh exact-sigma RGF case has crossed the scheduler boundary and is genuinely running now.
- The job has not yet finished the sigma-extraction-plus-native-run payload, so no local `transport_result.json` / `transport_runtime_cert.json` has been retrieved yet.

### Still Unresolved

1. Fresh exact-sigma benchmark parity is still unmeasured.
   - No fresh local `transport_result.json` exists yet under `tmp/iter17_sigma_exact`.
   - I therefore still do not have the first exact-sigma conductance to compare against the matching live Kwant point.

2. The finished benchmark artifact set is still absent.
   - `tmp/iter17_sigma_exact/model_b/c/kwant/kwant_reference.json`
   - `tmp/iter17_sigma_exact/model_b/c/rgf/rgf_raw.json`
   - `tmp/iter17_sigma_exact/model_b/c/comparison_raw.json`
   - `tmp/iter17_sigma_exact/model_b/c/comparison_fit.json`
   - `tmp/iter17_sigma_exact/benchmark_summary.json`

3. The `>= 5x` speedup requirement remains unproven.
   - The exact-sigma benchmark is still in-flight, so there is no finished summary to evaluate.

4. The configured clean restart path is still not green.
   - The separate `wtec init --validate-cluster --prepare-cluster-tools --prepare-cluster-pseudos --prepare-cluster-python` path still fails on missing ScaLAPACK during remote SIESTA preparation.

### What The Debugger Should Do Next

1. Keep monitoring the fresh exact-sigma run already in flight:
   - PID `28667`
   - root `tmp/iter17_sigma_exact`
   - jobs `60009` and `60010`

2. As soon as the first exact-sigma case finishes, inspect:
   - `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/transport_result.json`
   - `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/transport_runtime_cert.json`
   - `tmp/iter17_sigma_exact/model_b/c/rgf/d01_em0p2/transport/primary/sigma_manifest.json`

3. Verify the runtime metadata for that first fresh result includes:
   - `full_finite_sigma_source == "kwant_exact"`

4. Compare that exact-sigma conductance directly against the matching live Kwant point.
   - This is the next decisive parity signal.

5. If a future overlap run fails on the RGF branch, verify that the concurrent Kwant job is now automatically cancelled instead of leaking a stale queue occupant.
   - That is the new behavioral contract introduced in this turn.

6. Keep the restart-path gap separate.
   - This turn fixed overlap cleanup and queue hygiene.
   - It did not solve the independent ScaLAPACK/SIESTA failure in the full restart contract.
