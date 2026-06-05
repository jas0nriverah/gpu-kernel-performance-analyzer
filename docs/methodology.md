# Methodology

## Benchmark timing

- Use CUDA events around kernel launches to measure kernel runtime (`runtime_ms`).
- Run warmup iterations before timed iterations.
- Run repeated timed samples and retain all raw samples in `timing_samples.csv`.
- Compute summary statistics per scenario (`mean`, `median`, `min`, `max`, `p95`, `stddev`, `cv`).
  The coefficient of variation (`cv = stddev / mean`) is a unitless stability signal; a small
  `cv` means the measurement is reproducible.
- Nsight Compute profiling overhead is not used for benchmark timing claims.

## Derived estimates

- `effective_bandwidth_GBps = bytes_moved / runtime_seconds / 1e9`
- `effective_GFLOPs = flops / runtime_seconds / 1e9`
- `arithmetic_intensity = flops / bytes_moved`
- `speedup_runtime = baseline_runtime_mean / optimized_runtime_mean` (e.g. tiled vs naive GEMM)

These are estimates derived from declared operation/byte counts and measured runtime.

## Roofline

`analyze full` plots each scenario's `(arithmetic_intensity, effective_GFLOPs)` operating
point on log-log axes. Hardware ceiling lines (compute roof, DRAM roof) are drawn only when
the user supplies real peak numbers via `--peak-gflops` / `--peak-bandwidth-gbps`; no
hardware peaks are assumed or fabricated.

## Scenario sweeps

Scenarios are expanded from `configs/benchmark_scenarios.yaml`:

- `kernel`
- `problem_sizes`
- `block_sizes`
- `warmups`
- `repeats`
- `verify`

## Verification

Kernel correctness checks are optional via `verify=true` scenarios. Verification results are recorded in `benchmark_summary.csv`.

## Profiler metrics

Profiler metrics are imported only from real Nsight Compute CSV output. Imported rows must map to an exact benchmark scenario by `kernel + problem_size + block_size` and include source provenance.
