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

Occupancy/cache/SM metrics are marked unavailable unless a future Nsight Compute integration provides them.
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
- `plots/runtime_vs_size.png`
- `plots/effective_gflops_vs_size.png`
- `REPORT.md`

## Optional Nsight Compute import

Detect availability:

```bash
python -m gpu_kernel_analyzer profile ncu-detect
```

Import profiler metrics from a normalized CSV (`kernel,problem_size,block_size,metric_name,metric_value`):

```bash
python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/run_mvp \
  --source-tool ncu \
  --source-file reports/ncu_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv path/to/ncu_metrics.csv
```

`ncu-import` will reject rows that do not map to an exact benchmark scenario (`kernel + problem_size + block_size`) or missing provenance metadata.

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

## TODO: Real-GPU validation pass

- Run full benchmark sweep with compiled CUDA binary on target NVIDIA GPU.
- Capture real profiler source artifacts (`ncu` raw output) and import with provenance fields.
- Record final demo artifacts from real run (not fixture output).
