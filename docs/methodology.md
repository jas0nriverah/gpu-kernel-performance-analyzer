# Measurement and evaluation methodology

The [research report](research-report.md) connects the complete experiment, results and threats to validity. This page defines the measurement contracts; [modeling](modeling.md) describes the predictive tasks and [power methodology](power_measurement.md) gives collection commands.

## Timing, correctness and units

CUDA events bracket kernel launches to measure `runtime_ms`. Warmups precede timed repetitions. Allocation, host/device copies and CPU reference checks lie outside that interval. Every timing sample is retained in `timing_samples.csv`; per-scenario mean, median, minimum, maximum, p95, standard deviation and coefficient of variation are reported. `CV = stddev / mean` flags within-run variability; a small CV does not prove repeatability across allocations or dates. Nsight Compute overhead is excluded from timing claims.

Scenario configurations specify kernel, problem sizes, block sizes, warmups, repeats and verification. The default compatibility suite contains 24 configurations; the scaling suite contains 74; the edge suite contains 18. The four-device scaling study retains three separate sweeps × 50 samples × 74 configurations per device. Every requested correctness check in these captures passed. Raw manifests are authoritative for exact settings.

Current verification uses seeded nonuniform inputs and checks every output, including reduction block partials. Small GEMMs use arbitrary dense inputs with CPU dot products; larger GEMMs use a structured nonuniform input and a closed-form reference. This reduces large-N input diversity while checking every output. The historical A100 archive used an older eight-diagonal-element GEMM check. [Oracle coverage](correctness.md) explains this distinction. Verification is configurable in general CLI use; the research captures requested it.

## Derived estimates and roofline

| Quantity | Definition | Units |
| --- | --- | --- |
| Effective bandwidth | Declared bytes / runtime seconds / 10⁹ | GB/s |
| Effective compute rate | Declared FLOPs / runtime seconds / 10⁹ | GFLOP/s |
| Arithmetic intensity | Declared FLOPs / declared bytes | FLOP/byte |
| Runtime speedup | Baseline runtime / candidate runtime | Unitless |

Declared counts describe logical work, not measured DRAM traffic. Both GEMM implementations declare `3*N*N*sizeof(float)` despite different global-load behavior; that estimate cannot establish actual cache traffic. Reduction timing excludes the final CPU aggregation. Custom FP32 GEMM results are not Tensor Core or cuBLAS results.

`analyze full` plots scenario arithmetic intensity and effective GFLOP/s on log axes. Compute/DRAM rooflines appear only when supplied through `--peak-gflops` and `--peak-bandwidth-gbps`; no peaks are fabricated. Profiler metrics come only from real Nsight Compute CSVs matched to exact kernel/size/block scenarios with provenance.

## Sustained power and energy

Eight configurations run for three randomized trials, with at least 15 seconds sustained execution per window. Power queries request 0.2-second spacing; query duration adds overhead. Steady-window rows exclude the first and last two seconds. Trapezoidal integration of consecutive observations gives joules; dividing by covered span gives time-weighted mean watts. Coverage and excessive-gap checks prevent extrapolating across missing boundaries. No row or outlier trimming follows capture.

Estimated joules per launch combines steady GPU watts with sustained whole-window launch throughput, including synchronization/launch overhead. It is not isolated kernel energy or whole-server energy. Sensor averaging and correlated rows limit temporal interpretation. GB10's integrated GPU has unsupported memory-clock/framebuffer fields: these remain unavailable; GPU sensor watts exclude CPU/total SoC power. See [sensor scope and primary documentation](dgx-spark-validation.md).

## Data integrity

Capture manifests retain run/device identity, source/build/configuration provenance and artifact hashes. The [SQLite warehouse](warehouse.md) checks all listed hashes, cross-file identities and physical values; recomputes steady summaries; and inserts a capture transactionally. Same-run identical input is idempotent; conflicting input fails. Optional missing sensors are preserved in raw JSON rather than imputed at ingestion. Present nonfinite/malformed values fail validation. Model preprocessing, where applicable, is a separate train-fitted operation.

## ML splits and target

The target is contemporaneous per-row GPU sensor power in watts. It is not future power forecasting, per-kernel attribution or a hardware counter. H100 plus original H200 supply 2,276 development rows. Row IDs, UUIDs, sessions, timestamps, trial IDs and target-derived summaries are excluded from predictive features.

Two separate objectives use the fixed twelve-candidate grid:

- **Unseen-device selection:** H100→H200 and reverse. Minimize equal-weight mean directional RMSE. Whole-trial diagnostics do not change this winner.
- **Known-device session selection:** hold out each of six `(GPU UUID, trial)` groups, retaining other trials from that GPU in training. Minimize mean six-fold RMSE; this is validation, not an independent final test.

Numeric imputation/scaling and categorical imputation/encoding fit only training-fold rows. Baseline global/workload means use training targets only, with global fallback for unknown workloads. Full-data refitting occurs only after the selected configuration is frozen. The saved bundle is reloaded and checked for prediction parity.

The independent H200 session begins after the known-device freeze, uses a known GPU and new run/session IDs, and retains all 1,153 steady rows across 24 windows. GH200/GB10 are external-device evaluations; no final-target fitting, selection changes or clipping follow their scores. A reproduction using already-known outcomes is explicitly retrospective. [Detailed chronology and metrics](model-generalization.md).

## Statistical interpretation

Published timing aggregates use the median of three separate run means. Their min/max range is descriptive, not a confidence interval. Comparison defaults to within-run median runtime; `--statistic mean` is available. Cross-device gates are ineligible and their ratios cannot isolate hardware effects when host/build/date differ.

RMSE, MAE and R² are computed over all retained final rows. Rows/windows are correlated; row counts are not independent sample sizes. One prospective session and two development GPUs limit generalization. Error quantiles, per-workload/trial scores and complete residuals accompany headline metrics so rare large errors remain visible. No uncertainty calibration or broad population inference is claimed.
