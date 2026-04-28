# Methodology

## Benchmark timing

- Use CUDA events around kernel launches to measure kernel runtime (`runtime_ms`).
- Run warmup iterations before timed iterations.
- Run repeated timed samples and retain all raw samples in `timing_samples.csv`.
- Compute summary statistics per scenario (`mean`, `median`, `p95`, `stddev`, `cv`).
- Nsight Compute profiling overhead is not used for benchmark timing claims.

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

Kernel correctness checks are optional via `verify=true` scenarios. Verification results are recorded in `benchmark_summary.csv`.

## Profiler metrics

Profiler metrics are imported only from real Nsight Compute CSV output. Imported rows must map to an exact benchmark scenario by `kernel + problem_size + block_size` and include source provenance.
