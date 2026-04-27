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

## 8) Optional Nsight import (if available)

Normalized CSV must include:

- `kernel,problem_size,block_size,metric_name,metric_value`

```bash
python -m gpu_kernel_analyzer profile ncu-import \
  --run-dir outputs/real_gpu_small \
  --source-tool ncu \
  --source-file reports/ncu_small_raw.csv \
  --metric-set default_profiler_set \
  --ncu-csv reports/ncu_small_normalized.csv
```

Re-validate after import:

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/real_gpu_small
```

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
