# GPU Kernel Performance Analyzer

CUDA/C++ benchmark harness plus Python CLI for reproducible GPU kernel timing artifacts.

Current scope:

- build and run CUDA kernels for benchmark sweeps
- capture raw timing samples and summary CSVs
- emit run metadata JSON with measured device info when available
- validate artifact schema and metric provenance
- generate analysis CSV, plots, and a markdown report
- support optional Nsight Compute metric import without blocking MVP workflows

## Implemented default metrics

- `runtime_ms` - measured with CUDA events
- `effective_bandwidth_GBps` - derived estimate
- `effective_GFLOPs` - derived estimate
- `arithmetic_intensity` - derived estimate
- device metadata - measured from CUDA runtime when available

Profiler metrics are unavailable by default and only become measured for scenarios with imported Nsight Compute CSV rows.
The sweep manifest records whether `ncu` is detected on PATH, but MVP runs do not require or invoke Nsight Compute.

## Repository layout

- `benchmarks/` CUDA/C++ harness and kernels
- `configs/benchmark_scenarios.yaml` sweep definitions
- `configs/ncu_metric_sets.yaml` optional profiler import contract
- `src/gpu_kernel_analyzer/` Python CLI and validation logic
- `tests/` schema, parsing, heuristics, and CLI smoke tests

## Build benchmark binary

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

Prerequisite: CUDA Toolkit with `nvcc` available on PATH (or set `CUDAToolkit_ROOT` for CMake).

Expected binary:

- Windows: `build/Release/gpu_benchmark.exe` (or `build/gpu_benchmark.exe`)
- Linux/macOS: `build/gpu_benchmark`

## Run a sweep and write artifacts (real CUDA run)

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/Release/gpu_benchmark.exe \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run_mvp
```

## Validate artifacts

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/run_mvp
```

## Generate analysis + report

```bash
python -m gpu_kernel_analyzer analyze full --run-dir outputs/run_mvp
```

This writes:

- `analysis_heuristics.csv`
- `plots/runtime_vs_size_vector_reduction.png`
- `plots/runtime_vs_size_gemm.png`
- `plots/effective_gflops_vs_size_gemm.png`
- `plots/effective_bandwidth_vs_size_vector_reduction.png`
- `REPORT.md`

## Optional Nsight Compute import

Detect availability:

```bash
python -m gpu_kernel_analyzer profile ncu-detect
```

Print exact raw-capture/import commands for a selected scenario:

```bash
python -m gpu_kernel_analyzer profile ncu-plan \
  --binary build/gpu_benchmark \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --warmups 10 \
  --repeats 30 \
  --verify \
  --raw-csv-out outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv \
  --normalized-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --run-dir outputs/demo_real_gpu
```

Import profiler metrics from a normalized CSV (`kernel,problem_size,block_size,metric_name,metric_value`):

```bash
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv reports/ncu/vector_add_4194304_b256_raw.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set

python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/run_mvp \
  --source-tool ncu \
  --source-file reports/ncu_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv path/to/ncu_metrics.csv
```

`ncu-import` will reject rows that do not map to an exact benchmark scenario (`kernel + problem_size + block_size`) or missing provenance metadata.
Imported profiler metrics are scenario-specific and do not replace CUDA-event benchmark timing.

## Non-CUDA fixture demo (sample only)

This is a **sample/fixture** workflow for environments without `nvcc` or an NVIDIA GPU.
It is useful for CLI/schema/report validation, not for real performance claims.

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary tests/fixtures/fake_benchmark.py \
  --binary-interpreter python \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/sample_fixture

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/sample_fixture
python -m gpu_kernel_analyzer analyze full --run-dir outputs/sample_fixture
```

## Run tests

```bash
python -m pytest -q
```

## Notes

- If CUDA toolkit (`nvcc`) is unavailable, benchmark build will fail; artifact validation and analysis can still be tested with fixture-generated data.
- Occupancy/cache/SM metrics remain unavailable by default and only become measured after successful Nsight metric import.
- Real CUDA validation requires `nvcc` + NVIDIA GPU + successful benchmark binary build.

## Real GPU validation (A100 run)

- GPU: NVIDIA A100 80GB PCIe
- CUDA toolkit: 13.0
- Python: 3.11.9
- scenarios: 12
- validation: passed
- pytest: 17 passed
- Nsight Compute: run for two scenarios only (`vector_add` 4194304/256 and `gemm_tiled` 512/16)

Key result: tiled GEMM at `512x512` achieved about `3840` GFLOPs versus naive GEMM about `2589` GFLOPs, roughly `1.48x` faster.

Metric integrity for this run:

- `runtime_ms` is measured with CUDA events.
- `effective_bandwidth_GBps`, `effective_GFLOPs`, and `arithmetic_intensity` are derived estimates.
- Nsight profiler metrics are scenario-specific; unprofiled scenarios remain unavailable.
- Nsight timing overhead is not used for benchmark timing claims.

## TODO: Real-GPU validation pass

- Run full benchmark sweep with compiled CUDA binary on target NVIDIA GPU.
- Capture real profiler source artifacts (`ncu` raw output) and import with provenance fields.
- Record final demo artifacts from real run (not fixture output).
