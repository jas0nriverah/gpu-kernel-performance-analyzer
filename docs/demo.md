# Demo Guide

## Non-CUDA Fixture Demo

Use this workflow when CUDA Toolkit or an NVIDIA GPU is unavailable. It validates CLI behavior, artifact schema checks, analysis, plotting, and report generation with fixture data.

Fixture outputs are sample-only and must not be presented as real GPU performance results.

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

## Real CUDA Demo Requirements

- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- CMake
- Built benchmark binary

Build:

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

Run a real CUDA sweep:

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/demo_real_gpu

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

## Optional Nsight Compute Subset Pass

The Nsight Compute workflow is intentionally scenario-specific. It does not require rerunning a full benchmark sweep, and profiler timings are not used for benchmark `runtime_ms` claims.

Target scenarios used in the A100 validation snapshot:

1. `vector_add`, `problem_size=4194304`, `block_size=256`
2. `gemm_tiled`, `problem_size=512`, `block_size=16`

Check metric names on the target environment:

```bash
ncu --version
ncu --query-metrics | grep -E "sm__warps_active\\.avg\\.pct_of_peak_sustained_active|sm__throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__dram_throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__compute_memory_throughput\\.avg\\.pct_of_peak_sustained_elapsed|lts__throughput\\.avg\\.pct_of_peak_sustained_elapsed"
```

Print exact commands for a selected scenario:

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

Run Nsight Compute raw capture:

```bash
ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  > outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv
```

Normalize raw Nsight CSV rows:

```bash
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set
```

Import normalized profiler metrics:

```bash
python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/demo_real_gpu \
  --source-tool ncu \
  --source-file vector_add_4194304_b256_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv
```

Validate and regenerate the report:

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

Do not commit raw Nsight output by default. Keep profiler CSVs under ignored output paths unless intentionally adding a small labeled sample.
# Demo Guide

## Non-CUDA Fixture Demo

Use this workflow when CUDA Toolkit or an NVIDIA GPU is unavailable. It validates CLI behavior, artifact schema checks, analysis, plotting, and report generation with fixture data.

Fixture outputs are sample-only and must not be presented as real GPU performance results.

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

## Real CUDA Demo Requirements

- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- CMake
- Built benchmark binary

Build:

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

Run a real CUDA sweep:

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/demo_real_gpu

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

## Optional Nsight Compute Subset Pass

The Nsight Compute workflow is intentionally scenario-specific. It does not require rerunning a full benchmark sweep, and profiler timings are not used for benchmark `runtime_ms` claims.

Target scenarios used in the A100 validation snapshot:

1. `vector_add`, `problem_size=4194304`, `block_size=256`
2. `gemm_tiled`, `problem_size=512`, `block_size=16`

Check metric names on the target environment:

```bash
ncu --version
ncu --query-metrics | grep -E "sm__warps_active\\.avg\\.pct_of_peak_sustained_active|sm__throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__dram_throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__compute_memory_throughput\\.avg\\.pct_of_peak_sustained_elapsed|lts__throughput\\.avg\\.pct_of_peak_sustained_elapsed"
```

Print exact commands for a selected scenario:

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

Run Nsight Compute raw capture:

```bash
ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  > outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv
```

Normalize raw Nsight CSV rows:

```bash
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set
```

Import normalized profiler metrics:

```bash
python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/demo_real_gpu \
  --source-tool ncu \
  --source-file vector_add_4194304_b256_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv
```

Validate and regenerate the report:

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

Do not commit raw Nsight output by default. Keep profiler CSVs under ignored output paths unless intentionally adding a small labeled sample.
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
- pytest: 22 passed (current suite; 17 at the original A100 validation snapshot)
- Nsight Compute: run only for `vector_add` (`4194304`, `256`) and `gemm_tiled` (`512`, `16`)

Key result: tiled GEMM at `512x512` achieved about `3840` GFLOPs versus naive GEMM about `2589` GFLOPs, roughly `1.48x` faster.

Metric integrity notes:

- `runtime_ms` is measured with CUDA events.
- `effective_bandwidth_GBps`, `effective_GFLOPs`, and `arithmetic_intensity` are derived estimates.
- profiler metrics are scenario-specific; unprofiled scenarios remain unavailable.
- Nsight timing overhead is not used for benchmark `runtime_ms` claims.
- `occupancy` is treated as achieved occupancy (active warps percentage from `sm__warps_active.avg.pct_of_peak_sustained_active`), not theoretical occupancy limit.
- `l2_cache_hit_rate` remains unavailable unless directly measured and imported from `lts__t_sector_hit_rate.pct`.

## Optional Nsight Compute subset pass (PACE ICE)

This pass is intentionally small and scenario-specific. It does not require rerunning the full sweep.

Target scenarios:

1. `vector_add`, `problem_size=4194304`, `block_size=256`
2. `gemm_tiled`, `problem_size=512`, `block_size=16`

### Step 1: check Nsight metric names on target environment

```bash
ncu --version
ncu --query-metrics | grep -E "sm__warps_active\\.avg\\.pct_of_peak_sustained_active|sm__throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__dram_throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__compute_memory_throughput\\.avg\\.pct_of_peak_sustained_elapsed|lts__throughput\\.avg\\.pct_of_peak_sustained_elapsed"
```

### Step 2: print exact commands from the project helper

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

python -m gpu_kernel_analyzer profile ncu-plan \
  --binary build/gpu_benchmark \
  --kernel gemm_tiled \
  --problem-size 512 \
  --block-size 16 \
  --warmups 5 \
  --repeats 15 \
  --verify \
  --raw-csv-out outputs/demo_real_gpu/ncu_raw_gemm_tiled_512_16.csv \
  --normalized-csv outputs/demo_real_gpu/ncu_normalized_gemm_tiled_512_16.csv \
  --run-dir outputs/demo_real_gpu
```

### Step 3: run Nsight for each selected scenario

```bash
ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  > outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv

ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel gemm_tiled --problem-size 512 --block-size 16 --warmups 5 --repeats 15 --verify \
  > outputs/demo_real_gpu/ncu_raw_gemm_tiled_512_16.csv
```

### Step 4: normalize raw Nsight CSV rows

Normalize each raw file:

```bash
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set

python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv outputs/demo_real_gpu/ncu_raw_gemm_tiled_512_16.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_gemm_tiled_512_16.csv \
  --kernel gemm_tiled \
  --problem-size 512 \
  --block-size 16 \
  --metric-set default_profiler_set
```

Normalized CSV contract:

- `kernel,problem_size,block_size,metric_name,metric_value`
- metric names may include: `occupancy`, `sm_utilization`, `memory_throughput_pct`, `l2_throughput_pct`, `l2_cache_hit_rate`

### Step 5: import into existing real run artifacts

```bash
python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/demo_real_gpu \
  --source-tool ncu \
  --source-file vector_add_4194304_b256_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv

python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/demo_real_gpu \
  --source-tool ncu \
  --source-file gemm_tiled_512_b16_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/demo_real_gpu/ncu_normalized_gemm_tiled_512_16.csv
```

### Step 6: validate and regenerate report

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

Do not commit raw Nsight output by default. Keep it under ignored `outputs/` unless you intentionally add a very small, clearly labeled sample.
