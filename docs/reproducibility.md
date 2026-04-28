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

Linux:

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run_mvp
```

Windows:

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

## Optional Nsight Compute import

Profiler metrics are optional and scenario-specific. Normalize raw Nsight CSV output before import:

```bash
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv reports/ncu/vector_add_4194304_b256_raw.csv \
  --out-csv outputs/run_mvp/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set

python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/run_mvp \
  --source-tool ncu \
  --source-file vector_add_4194304_b256_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/run_mvp/ncu_normalized_vector_add_4194304_256.csv
```

Nsight Compute timings are not used for benchmark `runtime_ms` claims.

## Manifest guarantees

`run_manifest.json` includes:

- run id and UTC timestamp
- benchmark binary path and SHA-256
- scenario file path and SHA-256
- git commit when available
- runtime environment information
- optional Nsight Compute detection/import state

## Real-GPU limitation

If `nvcc` and an NVIDIA GPU are unavailable, use the fixture demo in `docs/demo.md` for non-CUDA workflow validation only.
Do not treat fixture outputs as real benchmark evidence.

See `docs/demo.md` for demo workflows and `docs/real_gpu_validation.md` for the real A100 validation snapshot.
