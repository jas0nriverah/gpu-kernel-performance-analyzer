# GPU Kernel Performance Analyzer

Benchmarks CUDA kernels with CUDA-event timing, derives bandwidth / GFLOPs / arithmetic
intensity, classifies each kernel as memory- or compute-bound, and optionally overlays
real Nsight Compute profiler metrics. Every number is tagged by how it was produced, so
estimates are never presented as measurements.

It also includes a small performance-prediction model and a tuning advisor built on top
of the measured data: predict a kernel's runtime for a configuration it has not run, rank
candidate block sizes, and generate plain-language tuning recommendations.

```text
CUDA kernels -> benchmark harness -> JSON/CSV artifacts -> analysis -> reports, plots, model
                (CUDA-event timing)   (per-run, validated)  (bandwidth/GFLOPs/roofline)
```

## What this demonstrates

- CUDA/C++ kernels and a timing harness, with Python for orchestration, analysis, plotting, and reporting.
- Honest measurement: every metric is tagged `measured`, `derived_estimate`, `predicted`, or `unavailable`, and the validator refuses to mark profiler metrics as measured unless real Nsight data was imported.
- ML on the project's own data: a runtime/bandwidth model and block-size autotuning, with leave-one-out cross-validation and an explicit small-data warning.
- Runs without a GPU: the full Python layer (analysis, model, advisor) is tested on CPU-only CI using committed sample artifacts and a fake benchmark binary.

## Supported kernels

| Kernel | Computes | Behavior it shows |
|--------|----------|-------------------|
| `vector_add` | `c[i] = a[i] + b[i]` | Classic memory-bound kernel; effective bandwidth is the headline metric. |
| `memcpy_bandwidth` | `out[i] = in[i]` | Pure streaming copy; closest single-kernel proxy for achievable DRAM bandwidth. |
| `stencil_1d` | 3-point weighted stencil | Modest data reuse and boundary handling; between copy and GEMM, still memory-bound. |
| `reduction` | block-wise tree sum | Shared memory, `__syncthreads()`, and the tree-reduction pattern. |
| `gemm_naive` | dense `C = A*B` | Compute-heavy but un-tiled; re-reads global memory. |
| `gemm_tiled` | dense `C = A*B`, 16x16 tiles | Shared-memory tiling that cuts DRAM traffic; the optimized baseline for the speedup comparison. |

## Metrics and provenance

Every value carries a provenance status so estimates are never confused with measurements.

| Metric | Status | Source |
|--------|--------|--------|
| `runtime_ms` | `measured` | CUDA events (`cudaEventElapsedTime`) |
| `effective_bandwidth_GBps` | `derived_estimate` | `bytes_moved / runtime` |
| `effective_GFLOPs` | `derived_estimate` | `flops / runtime` |
| `arithmetic_intensity` | `derived_estimate` | `flops / bytes_moved` |
| `device_metadata` | `measured` | CUDA runtime API |
| `occupancy`, `sm_utilization`, `memory_throughput_pct`, `l2_throughput_pct`, `l2_cache_hit_rate` | `unavailable` until imported, then `measured` | Nsight Compute CSV import only |
| model predictions | `predicted` | performance model, written to `model_predictions.csv` |

Nsight Compute timing overhead is never used for `runtime_ms`; CUDA-event timing is the
single source of truth. Profiler metrics only become `measured` for the exact scenarios
where a real Nsight CSV was imported. Predicted values live in their own artifact and are
never written into the measured run files; the validator rejects a `predicted` row if one
ever appears in the measured provenance.

## Quickstart

### 1. Build the CUDA harness (requires an NVIDIA GPU and `nvcc`)

```bash
cmake -S benchmarks -B build
cmake --build build --config Release
```

Binary: `build/gpu_benchmark` (Linux) or `build/Release/gpu_benchmark.exe` (Windows).
Set `CUDAToolkit_ROOT` if CMake cannot find the toolkit.

### 2. Run a benchmark sweep

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run
```

### 3. Validate artifacts (schema and metric policy)

```bash
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/run
```

### 4. Analyze: heuristics, speedup, plots, roofline, report

```bash
python -m gpu_kernel_analyzer analyze full --run-dir outputs/run \
  --peak-gflops <real_fp32_peak> --peak-bandwidth-gbps <real_dram_peak>
```

`--peak-*` are optional. If omitted, the roofline shows only your measured operating
points and no fabricated hardware ceilings. Supply your GPU's real peaks to draw the roofs.

### No GPU? Run the CPU-only fixture demo

This validates the full Python pipeline using a fake benchmark binary. Fixture outputs are
sample-only and must not be cited as real GPU performance.

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary tests/fixtures/fake_benchmark.py --binary-interpreter python \
  --scenarios configs/benchmark_scenarios.yaml --outdir outputs/sample_fixture
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/sample_fixture
python -m gpu_kernel_analyzer analyze full --run-dir outputs/sample_fixture
```

## Performance model and autotuning

A ridge regression in log space predicts `runtime_ms` and `effective_bandwidth_GBps` from
a configuration (kernel, problem size, block size). It uses scikit-learn when the optional
`[ml]` extra is installed and falls back to a NumPy solver otherwise; both produce the same
JSON model file. The intent is performance prediction and block-size autotuning, which are
standard problems in GPU performance engineering.

```bash
# Train from one or more run directories (more runs -> better model)
python -m gpu_kernel_analyzer model train \
  --run-dir outputs/run --model-out perf_model.json

# Predict runtime and bandwidth for a configuration without running it
python -m gpu_kernel_analyzer model predict \
  --model perf_model.json --kernel gemm_tiled --problem-size 512 --block-size 16

# Rank candidate block sizes by predicted runtime
python -m gpu_kernel_analyzer model recommend-block-size \
  --model perf_model.json --kernel vector_add --problem-size 4194304 \
  --candidates 64,128,256,512

# Rule-based tuning recommendations from a run's measured artifacts
python -m gpu_kernel_analyzer advise --run-dir outputs/run
```

Provenance and limitations:

- Predictions are tagged `predicted` and written to `model_predictions.csv`. They are never mixed into the measured benchmark artifacts.
- The model only learns from real runs you provide. `model train` accepts multiple `--run-dir` flags and pools them, so recommendations improve as you add benchmark runs.
- With few rows, `model train` prints a clear warning that cross-validation metrics are not statistically meaningful, and a block-size recommendation for a kernel whose training data did not vary block size is flagged as an extrapolation rather than a data-backed choice.
- The `advise` command is deterministic and offline. An optional `--llm` flag exists as a stub; no LLM dependency or API key is required for normal use or tests.

## Sample output

`analyze full` writes a self-contained run directory:

```text
outputs/run/
├── run_manifest.json            # git commit, device, env, scenario hashes, Nsight status
├── timing_samples.csv           # every raw CUDA-event sample
├── benchmark_summary.csv        # per-scenario mean/median/min/max/p95/stddev/cv + derived metrics
├── metrics_provenance.csv       # one row per metric with measured/derived/unavailable status
├── analysis_heuristics.csv      # memory- vs compute-bound classification
├── analysis_speedup.csv         # tiled vs naive GEMM speedup
├── plots/
│   ├── runtime_vs_size_gemm.png
│   ├── effective_gflops_vs_size_gemm.png
│   ├── effective_bandwidth_vs_size_vector_reduction.png
│   ├── effective_bandwidth_vs_size_memory_kernels.png
│   └── roofline.png
└── REPORT.md                    # human-readable summary of the run
```

### Real GPU validation snapshot (NVIDIA A100 80GB PCIe)

These are real measured numbers from a validation run, not estimates or examples. They
were captured with an earlier scenario set; the newer `memcpy_bandwidth` and `stencil_1d`
kernels have not yet been re-validated on a GPU (see Future work).

- GPU: NVIDIA A100 80GB PCIe, CUDA Toolkit 13.0, Python 3.11.9
- Artifact validation: passed. Nsight imported only for selected scenarios.

| Comparison (512x512 GEMM) | Effective GFLOPs (derived) |
|---------------------------|----------------------------|
| `gemm_tiled` | ~3840 |
| `gemm_naive` | ~2589 |
| Speedup (tiled vs naive) | ~1.48x |

Nsight Compute metrics for `vector_add` (size 4194304, block 256), imported from a real
profiler CSV: occupancy ~76.72%, SM utilization ~21.36%, memory throughput ~71.42%, L2
throughput ~77.37%. `l2_cache_hit_rate` stays `unavailable` because it was not captured. A
sanitized copy of this capture is committed at
`examples/ncu_raw_vector_add_4194304_b256.csv` for GPU-less demos and tests.

## Optional Nsight Compute workflow

```bash
# Detect Nsight Compute
python -m gpu_kernel_analyzer profile ncu-detect

# Print the exact raw-capture and import commands for one scenario
python -m gpu_kernel_analyzer profile ncu-plan --binary build/gpu_benchmark \
  --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  --raw-csv-out outputs/run/ncu_raw_vector_add.csv \
  --normalized-csv outputs/run/ncu_norm_vector_add.csv --run-dir outputs/run

# Normalize a raw Nsight CSV into import format (works on the committed sample)
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv examples/ncu_raw_vector_add_4194304_b256.csv \
  --out-csv outputs/run/ncu_norm_vector_add.csv \
  --kernel vector_add --problem-size 4194304 --block-size 256 --metric-set default_profiler_set

# Import normalized metrics (rejected unless they map to an exact benchmark scenario)
python -m gpu_kernel_analyzer profile ncu-import --run-dir outputs/run \
  --source-tool ncu --source-file ncu_raw_vector_add.csv \
  --metric-set default_profiler_set --ncu-csv outputs/run/ncu_norm_vector_add.csv
```

## Project structure

```text
benchmarks/                 CUDA/C++ harness
├── include/                kernel + harness headers
└── src/
    ├── benchmark_runner.cu CLI parsing + JSON output
    ├── cuda_utils.cu       device query + CUDA error checks
    └── kernels/            one .cu per kernel
configs/
├── benchmark_scenarios.{yaml,json}  sweep definitions
└── ncu_metric_sets.yaml             Nsight metric-set mappings
src/gpu_kernel_analyzer/    Python package
├── cli.py                  command-line interface
├── scenarios.py            config loading, expansion, validation + warnings
├── runner.py               invokes the benchmark binary
├── metrics.py              summary stats + derived metrics + provenance status
├── analysis.py             bottleneck heuristics + speedup
├── plotting.py             matplotlib plots + roofline
├── report.py               Markdown report generation
├── perf_model.py           performance prediction model + autotuning
├── advisor.py              rule-based tuning recommendations
├── ncu.py                  Nsight capture-plan / normalize / import
├── artifacts.py            artifact schema + provenance validation
└── ...
examples/                   sanitized sample Nsight raw CSV
tests/                      pytest suite (CPU-only) + fixtures
docs/                       methodology, metrics policy, reproducibility
```

## Design notes

- CUDA-event timing: `cudaEvent` timestamps are recorded on the GPU stream, so they measure device-side kernel execution without host-launch and CPU-clock noise. Wall-clock timers around an async launch mostly measure launch overhead.
- Warmups and repeated trials: the first launches pay one-time costs (context/JIT, allocation, cold caches, clock ramp-up). Warmups absorb those, and repeated timed trials give a distribution (min/median/mean/p95) plus a coefficient of variation that signals whether a result is stable enough to trust.
- Profiler overhead: Nsight Compute serializes and replays kernels, so its timing is not representative. Nsight is used only for hardware counters and never overrides the CUDA-event `runtime_ms`; the validator enforces this.
- Tiled vs naive GEMM: naive GEMM re-reads A and B from global memory for every output element. Tiled GEMM stages 16x16 tiles in shared memory so each loaded value is reused across a tile, cutting DRAM traffic and raising arithmetic intensity, which is why effective GFLOPs go up.
- Derived metrics: each kernel declares the bytes it moves and FLOPs it performs, then `bandwidth = bytes/runtime`, `GFLOPs = flops/runtime`, `intensity = flops/bytes`. These are labeled `derived_estimate`, never `measured`.

## Validating on a real GPU

On a machine with an NVIDIA GPU and the CUDA Toolkit, one script builds the harness,
smoke-tests the kernels with correctness checks, runs the full sweep, validates artifacts,
and runs the test suite:

```bash
pip install -e ".[dev]"
bash scripts/validate_gpu.sh
```

Optional roofline ceilings (supply your GPU's real peak FP32 and DRAM bandwidth):

```bash
PEAK_GFLOPS=19500 PEAK_BW_GBPS=1935 bash scripts/validate_gpu.sh
```

The script honors `BUILD_DIR`, `OUT_DIR`, `PYTHON`, and `SKIP_PIP_INSTALL` overrides. Set
`CUDAToolkit_ROOT` if CMake cannot locate the toolkit.

## Testing and CI

```bash
pip install -e ".[dev]"
python -m pytest -q          # 68 CPU-only tests
```

- Tests cover config parsing and validation, summary statistics, derived metrics, speedup, artifact-schema validation, report and plot generation, roofline, Nsight normalize/import, the performance model, autotuning, predicted-provenance separation, and the advisor.
- CI runs on CPU-only GitHub Actions across Python 3.10 to 3.12, plus a `ruff` lint job. It exercises everything except the CUDA kernels, using committed sample artifacts and a fake benchmark binary.
- The CUDA benchmarks require a CUDA-capable GPU and `nvcc` locally; they cannot run in standard GitHub Actions.
