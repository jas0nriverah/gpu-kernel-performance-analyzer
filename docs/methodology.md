# Methodology

## Benchmark timing

- Use CUDA events around kernel launches to measure kernel runtime (`runtime_ms`).
- Run warmup iterations before timed iterations.
- Run repeated timed samples and retain all raw samples in `timing_samples.csv`.
- Compute summary statistics per scenario (`mean`, `median`, `p95`, `stddev`, `cv`).

## Derived estimates

- `effective_bandwidth_GBps = bytes_moved / runtime_seconds / 1e9`
- `effective_GFLOPs = flops / runtime_seconds / 1e9`
- `arithmetic_intensity = flops / bytes_moved`

These are estimates derived from declared operation/byte counts and measured runtime.

## Scenario sweeps

Scenarios are expanded from `configs/benchmark_scenarios.yaml`:

- `kernel`
- `problem_sizes`
- `block_sizes`
- `warmups`
- `repeats`
- `verify`

## Verification

Kernel correctness checks are optional via `verify=true` scenarios. Verification failures are recorded in `benchmark_summary.csv`.
