# Repository integration review

Reviewed September 22, 2026. Analyzer starting revision: `6135e04`. Power-modeling source: `6d99cf7` from https://github.com/jas0nriverah/gpu-power-modeling. The saved “Polish backend SWE repo” conversation was read for its requirements: evidence-backed results, reliability, concise documentation, and a proper SVG architecture diagram.

## Feasibility and implementation

The projects are complementary and now share one installable repository. `gpu_kernel_analyzer` retains scenario execution, timing, regression comparison, and profiler analysis. `gpu_power_pipeline` adds telemetry normalization, model evaluation, persistence, predictions, serving, and monitoring. Both CLI names are preserved. Power ML/API dependencies are optional; base timing and measured-power capture stay lightweight.

This is a source integration on a separate branch, with upstream attribution and MIT license preserved under `third_party/gpu-power-modeling`. The separate GitHub repositories have not been renamed, archived, deleted, or pushed to. Original Git history has not been combined. The original power source remains independently available.

Baseline validation before integration:

| Project | Checks |
| --- | --- |
| Kernel analyzer | 90 tests passed; fresh CUDA Release build; 24 H100 scenarios and 18 boundary cases passed artifact validation |
| Power pipeline | 34 tests passed, including model persistence and FastAPI tests; installed with analyzer dependencies in a shared Python 3.11 environment |

After integration: **136 tests passed**, lint passed, the fresh 24-scenario and 18-boundary GPU checks passed again, and all **24 sustained H100 power windows** passed correctness checks. Real-telemetry grouped model evaluation and saved-model API/batch inference passed. Two upstream FastAPI/Starlette deprecation warnings remain in tests.

## Changes needed to make the integration meaningful

1. Preserve telemetry session IDs, workload labels, GPU UUID/name, and run/scenario/trial identity. The upstream loader replaced session/workload labels and dropped identity columns.
2. Exclude identifiers and split keys from predictive features. Reject missing/invalid time or grouped split inputs instead of falling back to random splitting.
3. Add sustained CUDA workloads with explicit start/end windows. Sensor readings cannot be meaningfully assigned to individual microsecond launches.
4. Capture measured board watts, raw query timing, idle history, correctness outputs, and hashes. Estimate energy only within covered intervals; keep predictions separate.
5. Preserve CPU CI, optional dependencies, and existing timing/regression behavior.

## Limits still worth addressing

- One H100 and three randomized trials do not establish cross-device generalization or thermal equilibrium.
- HIP/ROCm execution and AMD SMI telemetry remain planned, with no MI210 validation claimed.
- The original BMC path does target-correlation filtering before train/test splitting. Move learned feature filtering inside training folds before relying on that path for formal generalization claims.
- The model registry remains a local file-based registry; it has no concurrent-writer transaction protocol. Model selection and reported holdout performance are exploratory, not nested model-selection evaluation.
- The upstream README referenced historical metrics without committed raw output directories. Those tables are retained only in the archived upstream guide; the new H100 report is backed by committed raw evidence.

The next hardware campaign is specified in [GPU validation plan](gpu_validation_plan.md). Measured-power commands, scope, and model validation are documented in [power measurement](power_measurement.md).
