# Demo Guide

This guide shows two workflows: a CPU-only fixture demo that needs no GPU, and the real
CUDA workflow (including the optional Nsight Compute subset pass).

## CPU-only fixture demo (no GPU required)

This validates CLI behavior, artifact schema, analysis, plotting, and report generation
using a fake benchmark binary.

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

> Important: outputs generated from `tests/fixtures/fake_benchmark.py` are sample fixture
> outputs and must not be presented as real CUDA benchmark results.

## Real CUDA demo

Requirements:

- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- CMake
- Built benchmark binary

Build:

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

Run a real CUDA sweep, validate, and analyze:

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/demo_real_gpu

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

Or run the whole build → smoke-test → sweep → validate → analyze → test pipeline with one
script:

```bash
bash scripts/validate_gpu.sh
```

## Optional Nsight Compute subset pass

The Nsight Compute workflow is intentionally scenario-specific. It does not require
rerunning a full benchmark sweep, and profiler timings are never used for benchmark
`runtime_ms` claims.

Target scenarios (the ones used in the A100 validation snapshot):

1. `vector_add`, `problem_size=4194304`, `block_size=256`
2. `gemm_tiled`, `problem_size=512`, `block_size=16`

### Step 1: check Nsight metric names on the target environment

```bash
ncu --version
ncu --query-metrics | grep -E "sm__warps_active\.avg\.pct_of_peak_sustained_active|sm__throughput\.avg\.pct_of_peak_sustained_elapsed|gpu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed|gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed|lts__throughput\.avg\.pct_of_peak_sustained_elapsed"
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
```

### Step 3: run Nsight Compute raw capture

```bash
ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  > outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv
```

### Step 4: normalize raw Nsight CSV rows

```bash
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv \
  --out-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv \
  --kernel vector_add \
  --problem-size 4194304 \
  --block-size 256 \
  --metric-set default_profiler_set
```

Normalized CSV contract:

- columns: `kernel,problem_size,block_size,metric_name,metric_value`
- metric names may include: `occupancy`, `sm_utilization`, `memory_throughput_pct`, `l2_throughput_pct`, `l2_cache_hit_rate`

### Step 5: import normalized profiler metrics

```bash
python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/demo_real_gpu \
  --source-tool ncu \
  --source-file ncu_raw_vector_add_4194304_256.csv \
  --metric-set default_profiler_set \
  --ncu-csv outputs/demo_real_gpu/ncu_normalized_vector_add_4194304_256.csv
```

### Step 6: validate and regenerate the report

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

> Do not commit raw Nsight output by default. Keep profiler CSVs under ignored `outputs/`
> paths unless you intentionally add a small, clearly labeled sample.
