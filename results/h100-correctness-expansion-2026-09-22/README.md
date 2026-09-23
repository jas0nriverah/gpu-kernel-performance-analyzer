# Correctness validation record

Validated with `/tmp/gpu-expansion-build/gpu_benchmark` built for SM_90, on NVIDIA H100 80GB HBM3 (`GPU-91e942dc-edbb-7f2a-9cea-167356bbd86a`).

- All 24 configurations from `configs/benchmark_scenarios.yaml` passed.
- All 18 configurations from `configs/correctness_edges.json` passed.
- `gemm_naive` and `gemm_tiled` at 2048² passed; each ran one warmup and two timed repetitions.
- `vector_add`, `memcpy_bandwidth`, `stencil_1d`, and `reduction` at 67,108,864 elements passed; each ran one warmup and two timed repetitions.
- Direct CLI requests for zero size, reduction block 96, block 2048, excessive grid size, and GEMM signed-index overflow all failed before allocation or kernel work.
- `--device-info` reported the same UUID as `nvidia-smi`.

Per-run worker JSON and rejected-request stderr are stored beside this file. These timings are smoke validation only, not a performance comparison.

See [oracle coverage and launch checks](../../docs/correctness.md) for methodology and limitations.
