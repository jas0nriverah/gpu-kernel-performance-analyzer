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
cmake -S benchmarks -B build -DCMAKE_BUILD_TYPE=Release
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

See `docs/demo.md` for demo workflows and `docs/real_gpu_validation.md` for the H100 study and historical A100 comparison.

## Repeat the expanded cross-GPU study

Install the package first. The published H100 capture used architecture 90; use the matching architecture for your target GPU (the A100 is 80). Keep the source revision, optimization policy, configurations, and trial count consistent across machines.

```bash
PYTHON=python CUDA_ARCHITECTURES=90 OUT_DIR=outputs/h100 \
  bash scripts/cross_validate_gpu.sh
```

This builds in Release mode, captures tool versions/dependencies and GPU telemetry, runs the 24-scenario baseline, repeats the 74-scenario suite three times, executes 18 boundary checks, validates each artifact directory, and generates aggregate tables/charts. Run on an otherwise idle GPU; the script does not reserve the device or change clocks. It refuses an existing output directory. Separate captures for Nsight counters are described in the [published profiler notes](../results/h100-2026-09-22/profiler/README.md).

The published capture was built explicitly with:

```bash
cmake -S benchmarks -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --config Release -j 4
```

To regenerate the publication artifacts without running GPU kernels:

```bash
python scripts/summarize_results.py --results results/h100-2026-09-22
python -m gpu_kernel_analyzer compare \
  --baseline results/a100-2026-04-28 \
  --candidate results/h100-2026-09-22/baseline \
  --allow-device-mismatch --statistic mean --outdir outputs/a100-h100
```

`results/h100-2026-09-22/requirements-lock.txt` records the captured Python dependencies. Install the repository separately after installing those requirements if recreating that environment. Current run manifests also record source SHA-256 values. The published H100 base commit had local changes; see [capture provenance](../results/README.md), including the CUDA source patch. Absolute paths in historical manifests are capture-time provenance, not paths required on your machine.
