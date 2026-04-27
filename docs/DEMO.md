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

## TODO: Real-GPU validation pass

- Build and run `benchmarks/gpu_benchmark` on target GPU.
- Collect real `ncu` source artifact and import with provenance flags (`source_tool`, `source_file`, `metric_set`).
- Regenerate final report/plots from real benchmark artifacts.
