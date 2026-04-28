# GPU Kernel Performance Analyzer
CUDA/C++ benchmark harness plus Python CLI for reproducible GPU kernel timing, artifact validation, analysis, plotting, and profiler-backed metric import.
## Current Scope
- Build and run CUDA kernels for benchmark sweeps
- Measure kernel runtime with CUDA events
- Capture raw timing samples and summary CSVs
- Emit run metadata JSON with device and environment information
- Validate artifact schema and metric provenance
- Generate analysis CSVs, plots, and Markdown reports
- Optionally import scenario-specific Nsight Compute profiler metrics with provenance checks
## Implemented Metrics
Default metrics:
- `runtime_ms` - measured with CUDA events
- `effective_bandwidth_GBps` - derived estimate from bytes moved and measured runtime
- `effective_GFLOPs` - derived estimate from FLOP count and measured runtime
- `arithmetic_intensity` - derived estimate from FLOPs / bytes moved
- device metadata - measured from CUDA runtime when available
Profiler metrics are unavailable by default and only become measured for scenario/metric rows imported from real Nsight Compute CSV output.
Profiler-backed metrics currently supported:
- `occupancy` - achieved occupancy / active warps percentage from `sm__warps_active.avg.pct_of_peak_sustained_active`
- `sm_utilization`
- `memory_throughput_pct`
- `l2_throughput_pct`
- `l2_cache_hit_rate` - remains unavailable unless directly measured and imported from `lts__t_sector_hit_rate.pct`
Nsight Compute timing overhead is not used for benchmark timing claims. CUDA-event timing remains the source of `runtime_ms`.
## Repository Layout
- `benchmarks/` CUDA/C++ benchmark harness and kernels
- `configs/benchmark_scenarios.yaml` benchmark sweep definitions
- `configs/ncu_metric_sets.yaml` optional Nsight Compute metric set definitions
- `src/gpu_kernel_analyzer/` Python CLI, validation, analysis, plotting, reporting, and profiler import logic
- `tests/` CLI, schema, parsing, analysis, plotting, and Nsight import tests
- `docs/` demo, methodology, reproducibility, and metric policy notes
## Build Benchmark Binary
```bash
cmake -S benchmarks -B build
cmake --build build --config Release
Prerequisite: CUDA Toolkit with nvcc available on PATH, or set CUDAToolkit_ROOT for CMake.

Expected binary:

Windows: build/Release/gpu_benchmark.exe or build/gpu_benchmark.exe
Linux: build/gpu_benchmark
Run a Real CUDA Sweep
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run_mvp
On Windows, use the built .exe path if applicable:

python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/Release/gpu_benchmark.exe \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run_mvp
Validate Artifacts
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/run_mvp
Generate Analysis + Report
python -m gpu_kernel_analyzer analyze full --run-dir outputs/run_mvp
This writes:

analysis_heuristics.csv
plots/runtime_vs_size_vector_reduction.png
plots/runtime_vs_size_gemm.png
plots/effective_gflops_vs_size_gemm.png
plots/effective_bandwidth_vs_size_vector_reduction.png
REPORT.md
Optional Nsight Compute Workflow
Detect Nsight Compute:

python -m gpu_kernel_analyzer profile ncu-detect
Print exact raw-capture and import commands for a selected scenario:

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
Normalize raw Nsight CSV output:

python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv reports/ncu/vector_add_4194304_b256_raw.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set
Import normalized profiler metrics:

python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/demo_real_gpu \
  --source-tool ncu \
  --source-file vector_add_4194304_b256_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv
ncu-import rejects rows that do not map to an exact benchmark scenario using kernel + problem_size + block_size, or rows missing required provenance metadata.

Non-CUDA Fixture Demo
This workflow is for environments without nvcc or an NVIDIA GPU. It validates CLI behavior, artifact schemas, analysis, plotting, and reporting with fixture data.

Fixture outputs are not real GPU performance results.

python -m gpu_kernel_analyzer benchmark sweep \
  --binary tests/fixtures/fake_benchmark.py \
  --binary-interpreter python \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/sample_fixture
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/sample_fixture
python -m gpu_kernel_analyzer analyze full --run-dir outputs/sample_fixture
Run Tests
python -m pytest -q
Current validation: 22 passed.

Real GPU Validation
Validated on Georgia Tech PACE ICE.

GPU: NVIDIA A100 80GB PCIe
CUDA Toolkit: 13.0
Python: 3.11.9
Scenarios executed: 12
Artifact validation: passed
Test suite: 22 passed
Nsight Compute: imported only for selected scenarios
Nsight Compute profiler metrics were imported only for:

vector_add, problem_size=4194304, block_size=256
gemm_tiled, problem_size=512, block_size=16
Key Results
At 512x512:

gemm_tiled: about 3840 derived effective GFLOPs
gemm_naive: about 2589 derived effective GFLOPs
tiled GEMM was about 1.48x faster than naive GEMM
For vector_add, problem_size=4194304, block_size=256:

derived effective bandwidth: about 1329.6 GB/s
achieved occupancy / active warps percentage: about 76.72%
SM utilization: about 21.36%
memory throughput: about 71.42%
L2 throughput: about 77.37%
For gemm_tiled, problem_size=512, block_size=16:

achieved occupancy / active warps percentage: about 69.84%
SM utilization: about 58.63%
memory throughput: about 1.16%
L2 throughput: about 12.00%
l2_cache_hit_rate remains unavailable because it was not directly measured and imported.

Metric Integrity
runtime_ms is measured with CUDA events.
effective_bandwidth_GBps, effective_GFLOPs, and arithmetic_intensity are derived estimates.
Nsight profiler metrics are scenario-specific.
Unprofiled scenarios remain marked unavailable for profiler metrics.
Nsight timing overhead is excluded from benchmark timing claims.
Raw generated artifacts and profiler CSVs are kept out of git unless intentionally added as small labeled samples.
