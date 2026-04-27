# Reproducibility

## Environment

- Python 3.10+
- CUDA Toolkit with `nvcc`
- CMake 3.21+

Install package:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

## Build benchmark binary

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

## Run benchmark sweep

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/Release/gpu_benchmark.exe \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run_mvp
```

Scenario files can be YAML or JSON; this repo uses YAML as the primary config format.

## Validate artifacts

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/run_mvp
```

## Generate analysis outputs

```bash
python -m gpu_kernel_analyzer analyze full --run-dir outputs/run_mvp
```

## Manifest guarantees

`run_manifest.json` includes:

- run id and UTC timestamp
- benchmark binary path and SHA-256
- scenario file path and SHA-256
- git commit when available
- runtime environment information
- optional Nsight Compute detection/import state

## Real-GPU limitation

If `nvcc` and an NVIDIA GPU are unavailable, use the fixture demo in `docs/DEMO.md` for non-CUDA workflow validation only.
Do not treat fixture outputs as real benchmark evidence.
