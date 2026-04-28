# Real GPU Validation Checklist

Run these commands on a machine with NVIDIA GPU + CUDA toolkit.

## 1) Environment sanity

```bash
nvcc --version
nvidia-smi
```

## 2) Install project

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"
```

## 3) Build benchmark binary

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

Expected binary:

- Windows: `build/Release/gpu_benchmark.exe` (or `build/gpu_benchmark.exe`)
- Linux/macOS: `build/gpu_benchmark`

## 4) Small benchmark sweep

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/Release/gpu_benchmark.exe \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/real_gpu_small
```

## 5) Medium benchmark sweep

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/Release/gpu_benchmark.exe \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/real_gpu_medium
```

## 6) Validate artifacts

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/real_gpu_small
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/real_gpu_medium
```

## 7) Generate analysis outputs

```bash
python -m gpu_kernel_analyzer analyze full --run-dir outputs/real_gpu_small
python -m gpu_kernel_analyzer analyze full --run-dir outputs/real_gpu_medium
```

## 8) Optional Nsight Compute subset pass (no full sweep rerun required)

Capture only two scenarios for a small, defensible profiler pass:

- `vector_add`, `problem_size=4194304`, `block_size=256`
- `gemm_tiled`, `problem_size=512`, `block_size=16`

Check metrics on target machine:

```bash
ncu --version
ncu --query-metrics | grep -E "sm__warps_active\\.avg\\.pct_of_peak_sustained_active|sm__throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__dram_throughput\\.avg\\.pct_of_peak_sustained_elapsed|gpu__compute_memory_throughput\\.avg\\.pct_of_peak_sustained_elapsed|lts__throughput\\.avg\\.pct_of_peak_sustained_elapsed"
```

Optional helper to print exact commands:

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

Run Nsight raw capture for each selected scenario:

```bash
ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  > outputs/demo_real_gpu/ncu_raw_vector_add_4194304_256.csv

ncu --target-processes all --csv --page raw --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  build/gpu_benchmark --kernel gemm_tiled --problem-size 512 --block-size 16 --warmups 5 --repeats 15 --verify \
  > outputs/demo_real_gpu/ncu_raw_gemm_tiled_512_16.csv
```

Normalize each raw CSV:

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

Import normalized CSV files:

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

Re-validate and regenerate report:

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/demo_real_gpu
python -m gpu_kernel_analyzer analyze full --run-dir outputs/demo_real_gpu
```

Do not commit raw Nsight output unless intentionally included as a tiny, clearly labeled sample.

## Expected output folders

- `outputs/real_gpu_small`
  - `run_manifest.json`
  - `timing_samples.csv`
  - `benchmark_summary.csv`
  - `metrics_provenance.csv`
  - `analysis_heuristics.csv`
  - `REPORT.md`
  - `plots/`
- `outputs/real_gpu_medium`
  - same artifact structure as above
