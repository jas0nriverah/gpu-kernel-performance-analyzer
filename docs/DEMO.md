# Demo Guide

## Non-CUDA sample demo (fixture)

Use this when CUDA toolkit or NVIDIA GPU is unavailable.  
This validates CLI behavior, artifact schema, analysis, plotting, and reporting with fixture data.

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary tests/fixtures/fake_benchmark.py \
  --binary-interpreter python \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/sample_fixture

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/sample_fixture
python -m gpu_kernel_analyzer analyze full --run-dir outputs/sample_fixture
python -m gpu_kernel_analyzer profile ncu-detect
```

Important: outputs generated from `tests/fixtures/fake_benchmark.py` are sample fixture outputs and must not be presented as real CUDA benchmark results.

## Real CUDA demo requirements

- NVIDIA GPU available
- CUDA toolkit installed (`nvcc` available)
- benchmark binary built with CMake

## Real GPU validation (A100 run)

- GPU: NVIDIA A100 80GB PCIe
- CUDA toolkit: 13.0
- Python: 3.11.9
- scenarios: 12
- validation: passed
- pytest: 17 passed
- Nsight Compute: detected but not used

Key result: tiled GEMM at `512x512` achieved about `3840` GFLOPs versus naive GEMM about `2589` GFLOPs, roughly `1.48x` faster.

Metric integrity notes:

- `runtime_ms` is measured with CUDA events.
- `effective_bandwidth_GBps`, `effective_GFLOPs`, and `arithmetic_intensity` are derived estimates.
- occupancy/cache/SM metrics are unavailable because Nsight Compute was not run.
