# GPU Kernel Performance Analyzer

A reproducible CUDA benchmarking pipeline with a Python CLI, traceable measurement artifacts, and a performance regression gate. Run experiments, explain where time goes, and compare changes without losing the raw evidence.

[Results](docs/real_gpu_validation.md) · [Engineering decisions](docs/engineering.md) · [Compare runs](docs/comparisons.md) · [Reproduce](docs/reproducibility.md)

## Results you can inspect

Validated on **NVIDIA H100 80GB HBM3**, with recovered **A100 80GB PCIe** evidence for comparison. The H100 study covers **74 configurations × 3 independent sweeps × 50 samples = 11,100 raw timings**, plus a 24-scenario compatibility run and 18 boundary correctness cases. All requested correctness checks passed.

| H100 workload | Result | Evidence |
| --- | --- | --- |
| FP32 GEMM, 2048 × 2048 | Tiled: **8.16 TFLOP/s**, **1.51×** naive | Median of three run means |
| Vector add, 67.1M elements, 256 threads/block | **2.77 TB/s** effective bandwidth | Declared bytes / measured runtime |
| A100 → H100, tiled GEMM, 512 × 512 | **1.65×** observed speedup | 69.91 → 42.26 μs in matched scenarios |
| Repeatability across 74 configurations | Largest run-mean spread: **3.17%** | All three sweeps retained |

Cross-GPU ratios describe the captured runs: compiler versions, source revisions, and dates differ. Effective throughput is derived, not a hardware counter. Two configurations exceeded 10% within-run CV in at least one trial; the comparison report flags that noise. [Methodology, counters, and limitations →](docs/real_gpu_validation.md)

![H100 GEMM scaling and effective bandwidth across workload sizes](results/h100-2026-09-22/charts/scaling.png)

## How it works

![System architecture showing Python orchestration, artifact storage, offline analysis, and the CUDA execution boundary](docs/assets/system-flow.svg)

- **Bounded worker execution:** subprocess timeouts, checked JSON responses, exact scenario matching, and fail-fast correctness checks.
- **Reproducible evidence:** every timing sample, GPU metadata, binary/config/source hashes, and separate metric provenance.
- **Regression checks:** compare matching configurations, flag missing coverage and noisy measurements, export JSON/CSV/Markdown, and return CI exit codes.
- **Portable validation:** CPU-only fixtures exercise orchestration and analysis; CUDA runs validate kernel behavior separately.

The system uses local files and a CLI. [Design decisions and current limits →](docs/engineering.md)

## Try it without a GPU

Requires Python 3.10+. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# Rebuild the comparison from committed raw A100 and H100 evidence.
python -m gpu_kernel_analyzer compare \
  --baseline results/a100-2026-04-28 \
  --candidate results/h100-2026-09-22/baseline \
  --allow-device-mismatch --statistic mean \
  --outdir outputs/a100-h100
```

Open `outputs/a100-h100/REPORT.md` to inspect all 12 matching scenarios. No GPU is needed to compare or analyze existing runs. A [fixture demo](docs/demo.md) also exercises the full pipeline with explicitly synthetic timings.

## Run on your GPU

Requires an NVIDIA GPU, CUDA Toolkit (`nvcc`), and CMake 3.21+.

```bash
cmake -S benchmarks -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4

python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/run
python -m gpu_kernel_analyzer analyze full --run-dir outputs/run
```

Use a new output directory for each sweep. The default suite has 24 scenarios. For the larger repeated study, run `PYTHON=python bash scripts/cross_validate_gpu.sh`; see [reproducibility](docs/reproducibility.md) for build settings and the published capture commands.

## Gate a performance change

Capture `outputs/before` and `outputs/after` on the same GPU, then:

```bash
python -m gpu_kernel_analyzer compare \
  --baseline outputs/before --candidate outputs/after \
  --outdir outputs/regression --threshold-pct 5 \
  --fail-on-regression
```

The default statistic is median runtime. The gate fails for a slowdown over the threshold, excessive within-run variability, missing baseline coverage, or changed device/warmup/repeat settings. Cross-device comparisons require explicit opt-in and cannot pass a CI gate. [Policy and exit codes →](docs/comparisons.md)

## Workloads and measurement scope

| Kernel | What it exercises |
| --- | --- |
| `vector_add` | Three-array streaming access |
| `memcpy_bandwidth` | Device-to-device copy kernel |
| `stencil_1d` | Neighbor reads and boundary handling |
| `reduction` | Shared-memory tree reduction into block partial sums |
| `gemm_naive` | FP32 matrix multiplication with direct global reads |
| `gemm_tiled` | FP32 matrix multiplication using 16 × 16 shared-memory tiles |

CUDA events measure kernel execution; allocations, host/device copies, and CPU verification are outside the timed interval. Reduction timing excludes the final CPU aggregation. GEMM uses ordinary FP32 arithmetic, not Tensor Cores or cuBLAS. Simple deterministic inputs provide smoke checks, not exhaustive numerical validation.

Nsight Compute supplies optional hardware counters for exact scenarios. Model predictions stay in separate files. The optional ridge model and tuning advisor are exploratory; [usage and limitations](docs/modeling.md) are documented separately.

## Code and tests

```text
benchmarks/                CUDA kernels, event timing, JSON worker
src/gpu_kernel_analyzer/   CLI, runner, statistics, comparison, reporting
configs/                   Baseline, scaling, and boundary suites
results/                   Captured A100/H100 evidence and generated charts
scripts/                   GPU validation and result regeneration
tests/                    CPU tests and deterministic worker fixture
docs/                     Methodology, design, reproduction, and results
```

```bash
python -m pytest -q
python -m pip install -e ".[lint]"
ruff check .
```

GitHub Actions runs CPU tests on Python 3.10–3.12 and lint checks. The published H100 capture was validated locally; GPU performance is not tested by hosted CPU CI.

[MIT license](LICENSE)
