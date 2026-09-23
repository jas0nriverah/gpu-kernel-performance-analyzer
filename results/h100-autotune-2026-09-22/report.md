# Measured autotuning results

All rankings use measured sustained trials. Runtime/energy are per launch; workloads are grouped by mathematical operation and size.

## vector_add:67108864

Selected: `vector_add-n67108864-b256` (confirmation: **pass**)

| Candidate | Median runtime (s/launch) | Median energy (J/launch) | Runtime MAD | Energy MAD |
|---|---:|---:|---:|---:|
| vector_add-n67108864-b128 | 0.000319687 | 0.145 | 2.28e-08 | 0.000225 |
| vector_add-n67108864-b256 | 0.000288226 | 0.13758 | 6.04e-09 | 0.000582 |
| vector_add-n67108864-b512 | 0.000296508 | 0.137728 | 2.65e-08 | 6.09e-05 |

## Evidence and limits

Raw telemetry, trial summaries, worker stdout/stderr, manifests, preflight failures, and independent confirmation captures are retained in this output directory. Capture hashes are checked before ranking. Confirmation compares independent medians within 20%; this is a coarse repeatability screen, not a statistical confidence interval. GPU board power includes device-level circuitry and sensor averaging.
