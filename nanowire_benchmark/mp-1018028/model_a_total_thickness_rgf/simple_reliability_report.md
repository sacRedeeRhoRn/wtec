# Model A RGF Reliability Report

## Scope

This report summarizes the available `model_a` nanowire RGF artifacts on disk and uses the only direct Kwant-vs-RGF validation artifact currently available for this geometry family.

There are two distinct data sources:

1. `model_a_total_thickness_rgf`
   - real multi-thickness RGF scan
   - thicknesses `1..5`
   - energies `-0.2, -0.1, 0.0, 0.1, 0.2`
   - width `5`
   - length `24`
   - mode `full_finite`
   - `eta = 1e-6`
   - no staged exact Kwant self-energies in the payloads
   - source summary: `results_summary.json`, `results_summary.csv`

2. `tmp/model_a_single_t5_exactsigma_remote_run_v3`
   - direct Kwant-vs-RGF comparison for thickness `5` only
   - width `5`
   - length `24`
   - energies `-0.2, -0.1, 0.0, 0.1, 0.2`
   - exact-sigma validation path
   - source files: `target_summary.json`, `comparison.json`, `speed_summary.json`

## Geometry

- Material: `TiS`
- Model: `model_a`
- Axis: `c`
- Width: `5 uc`
- Length: `24 uc`
- Multi-thickness scan: `thickness_uc = 1, 2, 3, 4, 5`
- Energies: `-0.2, -0.1, 0.0, 0.1, 0.2`

## Multi-Thickness RGF Scan

These are the retrieved RGF transmissions from the completed remote scan in `model_a_total_thickness_rgf`.

| thickness_uc | G(E=-0.2) | G(E=-0.1) | G(E=0.0) | G(E=0.1) | G(E=0.2) | wall_s_min | wall_s_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.02284005328030027 | 0.515896001138677 | 0.2215836777105759 | 0.68791883361496 | 0.1827186362283914 | 2.728116 | 2.789312 |
| 2 | 0.852863556523798 | 0.0234515610646679 | 0.3977005709809477 | 0.5045140377915788 | 2.081993734762202 | 8.926891 | 9.165957 |
| 3 | 0.2697752078589881 | 1.039843184351647 | 0.340778326074424 | 0.2020737865804954 | 0.9442842179917527 | 23.736259 | 24.278072 |
| 4 | 0.9582102476548803 | 0.743054566208256 | 0.3382405945745651 | 0.0699750143736133 | 0.354669227661612 | 50.354711 | 51.439531 |
| 5 | 0.9000928717420199 | 0.1180036342934221 | 0.4016169335978155 | 0.1736005616876538 | 1.311100554103483 | 95.167519 | 101.857931 |

Interpretation:

- The RGF scan does produce stable outputs for all `25` requested `(thickness, energy)` points.
- Runtime increases sharply with thickness, from about `2.7 s` at `t=1` to about `95-102 s` at `t=5`.
- This scan alone does **not** verify reliability against Kwant, because no matching multi-thickness Kwant reference is present for these points.

## Direct Kwant Comparison Available On Disk

The only direct Kwant-vs-RGF comparison currently available for `model_a` is the exact-sigma thickness-`5` run in `tmp/model_a_single_t5_exactsigma_remote_run_v3`.

| energy_rel_fermi_ev | Kwant | RGF | abs_delta | rel_delta | rgf_wall_seconds |
| --- | --- | --- | --- | --- | --- |
| -0.2 | 10.00000000000215 | 0.8870683237829349 | 9.112931676219215 | 0.9112931676217256 | 14.590562 |
| -0.1 | 12.000000000000258 | 0.1939310757023255 | 11.806068924297932 | 0.9838390770248066 | 14.607701 |
| 0.0 | 12.00000000000041 | 0.004365055846856454 | 11.995634944153554 | 0.9996362453460953 | 14.627935 |
| 0.1 | 10.000000000000862 | 0.0576471509171212 | 9.94235284908374 | 0.9942352849082883 | 14.586855 |
| 0.2 | 5.999999999996259 | 0.0160526820046281 | 5.983947317991631 | 0.997324552999227 | 14.603662 |

Interpretation:

- This is a severe mismatch.
- Absolute error is roughly `6` to `12 e^2/h`.
- Relative error is roughly `91%` to `100%`.
- So for the validated exact-sigma `t=5` case, the current Model A RGF result is **not reliable** relative to Kwant.

## Simple Speed Report

### What is available

- Multi-thickness RGF scan wall times are available from the raw runtime certs.
- Exact-sigma `t=5` RGF wall times are available and total:
  - `rgf_total_wall_seconds = 73.016715`

### What is not available

- A matching Kwant wall clock for the exact-sigma `t=5` comparison is not present in the saved artifacts:
  - `kwant_wall_seconds = null`
  - `speedup_vs_kwant = null`

### Practical conclusion

- We can describe RGF runtime scaling for Model A.
- We **cannot** claim a Model A speed enhancement over Kwant from the currently saved artifacts, because the Kwant wall time is unavailable here.

## Bottom Line

- `model_a` RGF has been run for multiple thicknesses and energies.
- That multi-thickness scan is useful as an execution artifact, but it is **not** by itself a reliability proof.
- The only direct Kwant comparison currently on disk for `model_a` is the exact-sigma `t=5` case, and it fails badly.
- Therefore the current evidence says:
  - `model_a` RGF is runnable
  - `model_a` RGF is **not yet verified reliable** against Kwant
  - `model_a` speed enhancement vs Kwant is **not yet established**

## Source Files

- `nanowire_benchmark/mp-1018028/model_a_total_thickness_rgf/results_summary.json`
- `nanowire_benchmark/mp-1018028/model_a_total_thickness_rgf/results_summary.csv`
- `tmp/model_a_single_t5_exactsigma_remote_run_v3/target/target_summary.json`
- `tmp/model_a_single_t5_exactsigma_remote_run_v3/comparison.json`
- `tmp/model_a_single_t5_exactsigma_remote_run_v3/speed_summary.json`
