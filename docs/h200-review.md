# H200 validation — 20 October 2026

This review exercised the CUDA benchmark, measured-power capture, a fixed H100-trained model on H200, and saved-model serving. Raw evidence remains local under `outputs/prizepicks-h200-2026-10-20/` (ignored by Git); this document reports measurements rather than promising production performance.

## Device and provenance

NVIDIA H200, UUID `GPU-c3ce36e6-9e77-7ff0-521c-c3155f0a713b`, compute capability 9.0, 132 multiprocessors, 150,156,607,488 bytes global memory. Built with CUDA 12.6.1 for `sm_90`; Python 3.11.9. Benchmark binary SHA-256: `4c0b64dbb1a018d3b588dbe82792a038c38409b56d488922a65d0bca2c59e536`.

Manifests identify source commit `335069725968e031ed82c1e847df4853eb9c5976` and source hashes; this review ran with local uncommitted reporting changes. The commit identifier alone does not describe the working tree. The timing directory includes GPU snapshots. The authoritative final package freeze is `outputs/prizepicks-h200-2026-10-20/requirements-lock.txt`; the earlier timing-directory freeze may have overlapped dependency installation.

## CUDA timings

- Baseline: 24 configurations, 660 timing samples.
- Main sweep: 74 configurations × 3 independent runs × 50 repetitions = **11,100 timing samples**.
- Edge cases: 18 configurations, 90 timing samples. Every baseline, main-sweep, and edge-case result passed deterministic verification.
- Maximum effective throughput: **3,404.7 GB/s** for vector addition; **8,156.1 GFLOP/s** for tiled GEMM. Effective byte/FLOP formulas are workload estimates, not hardware counters or vendor-library results.
- Across the 74 configurations, median spread between trial means was **0.239%**, maximum **3.622%**. Six configurations had a maximum within-run coefficient of variation above 5%; maximum CV was **12.25%**. Small-kernel timing claims need that context.

The descriptive H100 comparison matched all 74 scenarios and found zero slowdowns exceeding 5% in the selected H200 run. Its CI gate correctly reports **not eligible** because GPU, source/build, and environment metadata differ. These ratios cannot isolate an H200 hardware effect or establish statistical significance. Full tables: `timing/AGGREGATE.md` and `h100-h200-comparison/REPORT.md` beneath the evidence root.

## Measured board power and external model evaluation

The capture completed **24 verified workload windows** (eight configurations × three randomized trials), **2,083 raw telemetry queries**, and **1,153 steady-window rows**. Each workload ran for at least 15 seconds; the first/last two seconds were trimmed. `summarize_power.py` verified every captured artifact hash, device UUID, trial count, and recomputed summary against raw telemetry before publishing `power/REPORT.md`.

At matched 2048 × 2048 FP32 GEMM, tiled GEMM delivered **1.51× sustained throughput** and **41.0% lower estimated energy per launch** than naive GEMM. Mean board power was 385.6 W versus 432.9 W. These compare the repository's custom kernels, without Tensor Cores or cuBLAS. Energy per launch combines steady board watts with whole-window launch throughput, including launch/synchronization overhead; it is not isolated kernel energy. GPU-board telemetry excludes full-server power. Correlated sensor readings and three trial ranges do not establish confidence intervals.

Linear regression was fixed from the published H100 study before inspecting H200 targets. It trained only on **1,123 H100 rows** and evaluated all **1,153 H200 rows**, with disjoint GPU UUIDs. No H200 fitting, tuning, clipping, or adaptation occurred. Both baselines use training targets only.

| Predictor | H200 MAE (W) | H200 RMSE (W) | H200 R² |
| --- | ---: | ---: | ---: |
| Fixed H100 linear regression | 50.506675 | 56.779318 | -0.791256 |
| H100 global mean | 38.659656 | 43.000354 | -0.027357 |
| H100 workload mean | 26.336629 | 29.486646 | 0.516910 |

**The trained model underperformed both baselines on H200.** This experiment demonstrates a reproducible external-device evaluation and exposes a generalization failure. H100 and H200 captures also differ in date and environment, so the result cannot isolate hardware as the cause. Samples within workloads are correlated and are not 1,153 independent trials.

Monitoring emitted **three warnings**: `memory_clock_mhz` mean shift and out-of-reference-range share (2619 MHz during H100 training versus 3201 MHz on H200), and unseen `gpu_name` (all H200 rows). The clock reference has zero variance; its standardized-shift magnitude is dominated by the implementation's numerical floor and should not be interpreted as a meaningful effect size. Complete feature checks are in `transfer/monitoring/drift_report.csv`.

Evidence: `transfer/evaluation.json` contains input/model/evaluator hashes and scores; `transfer/predictions.csv` and `transfer/per_workload.csv` retain row-level predictions and workload errors. `transfer/REPORT.md` presents the evaluation; `power/manifest.json`, raw CSVs, and `power/REPORT.md` retain capture provenance and measurement scope.

## Saved model and checks

**Historical, before API hardening:** the saved bundle was SHA-256 verified and reloaded. FastAPI's in-process TestClient returned HTTP 200 for `/health`, `/model`, and `/predict`; readiness reported the loaded linear regression version `20261020T201154Z`. All **1,153 feature-only records** produced finite API predictions, exactly matching direct predictions from the reloaded bundle (**0 W maximum absolute difference**). This validates saved-model serving parity, not network latency or load capacity. Evidence: `transfer/api-verification.json` is preserved as the historical permissive-serving result. The current API rejects all 1,153 H200 rows with HTTP 422 because the model reference profile contains only H100. Current H100 replay still matches direct predictions exactly. See [quality validation](quality-validation.md) and the current reproduction below.

The full test suite completed with **165 passed** in 101.44 seconds and one Starlette/httpx deprecation warning. The final focused transfer tests passed **8 tests** after coverage expansion; the earlier full-suite count predates that expansion. Ruff passed for the changed summary scripts, external evaluator, evaluator tests, and API replay script.

## Reproduction

Run from the repository root with the project virtual environment. Timing configurations are `configs/benchmark_scenarios.yaml`, `configs/cross_validation.yaml`, and `configs/correctness_edges.json`; the saved per-run manifests are the authority for exact scenarios and repetitions. The existing completed timings were reused during this review.

```bash
module load cuda/12.6.1
H200_RUN="outputs/h200-replay-$(date -u +%Y%m%dT%H%M%SZ)"
GPU_UUID=GPU-c3ce36e6-9e77-7ff0-521c-c3155f0a713b \
  PYTHON=.venv/bin/python MODE=full OUT_DIR="$H200_RUN/timing" \
  bash scripts/cross_validate_gpu.sh
.venv/bin/python -m gpu_kernel_analyzer compare \
  --baseline results/h100-2026-09-22/trial-1 \
  --candidate "$H200_RUN/timing/trial-1" \
  --allow-device-mismatch --statistic mean \
  --outdir "$H200_RUN/h100-h200-comparison"
.venv/bin/python -m gpu_kernel_analyzer power capture \
  --binary "$H200_RUN/timing/build/gpu_benchmark" \
  --scenarios configs/power_h100.yaml \
  --device GPU-c3ce36e6-9e77-7ff0-521c-c3155f0a713b \
  --duration-seconds 15 --trials 3 \
  --outdir "$H200_RUN/power"
.venv/bin/python scripts/summarize_power.py --run-dir "$H200_RUN/power"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 \
  .venv/bin/python -m gpu_power_pipeline.transfer \
  --train results/h100-power-2026-09-22/steady_telemetry.csv \
  --test "$H200_RUN/power/steady_telemetry.csv" \
  --outdir "$H200_RUN/transfer" --model linear_regression
.venv/bin/python scripts/verify_saved_api.py \
  --transfer-dir "$H200_RUN/transfer" \
  --telemetry "$H200_RUN/power/steady_telemetry.csv" \
  --expect-unseen-gpu --out "$H200_RUN/api-rejection.json"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 \
  .venv/bin/python -m pytest -q
```

Capture and transfer require fresh output directories to preserve prior evidence. Adjust paths together for a new run; the device UUID is specific to this allocation. Install the `dev,power,api,lint` extras before reproducing. The original review performed no network serving measurement. A later short localhost HTTP smoke is documented in [quality validation](quality-validation.md); no production capacity claim is made.
