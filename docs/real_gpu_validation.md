# Real GPU Validation

This page summarizes the real GPU validation snapshot for the project. Fixture outputs are excluded from these performance claims.

## Environment

- GPU: NVIDIA A100 80GB PCIe
- CUDA Toolkit: 13.0
- Python: 3.11.9
- Artifact validation: passed

This snapshot was captured with an earlier scenario set. The current default config
(`configs/benchmark_scenarios.yaml`) expands to 24 scenarios and the CPU-only test suite
has grown to 68 tests. The newer `memcpy_bandwidth` and `stencil_1d` kernels build and
pass local correctness checks but have not yet been folded into this measured A100
snapshot; their numbers will be added after a dedicated GPU run.

## Metrics Policy

- `runtime_ms` is measured with CUDA events.
- `effective_bandwidth_GBps`, `effective_GFLOPs`, and `arithmetic_intensity` are derived estimates.
- Nsight Compute metrics are scenario-specific and only measured for rows imported from real Nsight CSV output.
- Nsight Compute timing overhead is not used for benchmark timing claims.
- `occupancy` is interpreted as achieved occupancy / active warps percentage from `sm__warps_active.avg.pct_of_peak_sustained_active`, not theoretical occupancy.
- `l2_cache_hit_rate` remains unavailable unless directly measured and imported from `lts__t_sector_hit_rate.pct`.

## Validated Results

At `512x512`:

- `gemm_tiled`: about `3840` derived effective GFLOPs
- `gemm_naive`: about `2589` derived effective GFLOPs
- tiled GEMM was about `1.48x` faster than naive GEMM

For `vector_add`, `problem_size=4194304`, `block_size=256`:

- derived effective bandwidth: about `1329.6 GB/s`
- achieved occupancy / active warps percentage: about `76.72%`
- SM utilization: about `21.36%`
- memory throughput: about `71.42%`
- L2 throughput: about `77.37%`

For `gemm_tiled`, `problem_size=512`, `block_size=16`:

- achieved occupancy / active warps percentage: about `69.84%`
- SM utilization: about `58.63%`
- memory throughput: about `1.16%`
- L2 throughput: about `12.00%`

## Nsight Compute Coverage

Nsight Compute profiler metrics were imported only for:

1. `vector_add`, `problem_size=4194304`, `block_size=256`
2. `gemm_tiled`, `problem_size=512`, `block_size=16`

Unprofiled scenarios remain unavailable for profiler metrics.

## Reproduce the Workflow

See:

- `docs/reproducibility.md`
- `docs/demo.md`
- `docs/metrics_policy.md`
