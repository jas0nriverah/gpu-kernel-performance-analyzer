# Measured power and GPU efficiency

The integrated project combines CUDA kernel timing with sustained-workload board-power capture and an optional telemetry-to-power modeling pipeline. Both original CLIs remain available. The source import is documented in [ORIGIN.md](../third_party/gpu-power-modeling/ORIGIN.md), with its original MIT notice retained.

## Capture on an allocated NVIDIA GPU

Use Python 3.10+ (the current validation environment uses 3.11). The base analyzer needs no ML dependencies for power capture. Install `.[power,api]` for model training and serving.

```bash
python -m pip install -e '.[dev,power,api,lint]'
cmake -S benchmarks -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j 4
python -m gpu_kernel_analyzer power capture \
  --binary build/gpu_benchmark --scenarios configs/power_h100.yaml \
  --device 0 --duration-seconds 15 --trials 3 \
  --outdir outputs/h100-power
python scripts/summarize_power.py --run-dir outputs/h100-power
```

Select the allocated physical GPU by UUID when device indices are ambiguous. The capture selects that same UUID for CUDA and telemetry, checks for existing compute processes, and never changes clocks or power limits. It does not reserve the GPU; run inside an exclusive allocation. A new output directory is required. A failed capture retains raw evidence and a `failed` manifest; it cannot generate a completed results report.

Each randomized trial covers six kernels and eight configurations: three vector-add block sizes, copy, stencil, reduction, naive GEMM, and tiled GEMM. Every scenario requests correctness verification. Matrix size is 2048; streaming workloads use 67,108,864 elements. These fit the H100 used here; check memory capacity before extending the suite.

## What is measured

- **Timing:** CUDA events around individual launches, collected separately after sustained execution.
- **Board power:** `nvidia-smi --query-gpu=power.draw` with GPU UUID, utilization, clocks, and temperature. Host query midpoint and query duration are retained. The default waits 200 ms between queries, so actual cadence includes query overhead.
- **Sustained throughput:** completed launches divided by monotonic elapsed time for batches of 100 launches, including synchronization and launch overhead. Allocations, transfers, and correctness checks are outside the sustained window.
- **Energy:** trapezoidal integration of power over the covered, trimmed sample interval. Missing coverage, long gaps, invalid watts, or clock discontinuities fail validation. This is a derived estimate, not a hardware energy-counter reading.
- **Energy per launch:** steady mean watts divided by the whole-window sustained launch rate. This assumes that rate represents the steady interval. Treat it as an estimate for this sustained workload, not direct energy measurement of an isolated launch.

[NVIDIA's NVML documentation](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html) describes one-second power averaging on Ampere other than GA100 and newer architectures. H100 samples taken at 200 ms intervals are therefore correlated. The first and last two seconds of each sustained workload are excluded from power summaries. Sensor behavior differs across GPUs and must be recorded when comparing them.

The reported scope is the GPU board and associated circuitry, not the entire server or wall socket. Pre-workload idle periods retain thermal history and are reported separately; no idle subtraction is applied. Three trial ranges are descriptive, not confidence intervals. FP32 GEMM here uses custom scalar kernels, not Tensor Cores or vendor BLAS.

## Train and validate a power model

```bash
gpu-power-pipeline run --source local_nvidia_smi \
  --data-path outputs/h100-power/steady_telemetry.csv \
  --split-strategy grouped --run-validation --generate-report \
  --save-model --registry-dir outputs/power-registry \
  --outdir outputs/h100-power-model
```

`session_id` identifies the entire randomized trial. Grouped validation holds out a whole trial, including all workloads, rather than splitting adjacent sensor readings between train and test. It evaluates repeat-session behavior on **one H100**, not transfer to an unseen GPU or unseen workload. Random/time split diagnostics are secondary. Model inputs exclude trial/run/session IDs, UUIDs, timestamps, and source filenames. Workload labels and physical telemetry remain available as predictors.

The original power package also supports synthetic and public datasets, persistence, batch predictions, an optional FastAPI API, and drift reports. See the [archived upstream guide](../third_party/gpu-power-modeling/README.md) for command concepts; configurations now live under `configs/power/`, and installation uses the root extras. The upstream README's old metrics are not new validation evidence. BMC correlation-based feature filtering still occurs before splitting; treat that historical path as exploratory until feature selection is fitted only on training folds. The new local H100 path does not apply that target-correlation filter.

The measured H100 dataset and held-out-trial results are in the [power-model validation report](../results/h100-power-model-2026-09-22/REPORT.md).

## Reproduce and inspect

[The H100 report](../results/h100-power-2026-09-22/REPORT.md) links every telemetry row, per-trial summary, correctness result, and source/binary hash. Regenerate with `scripts/summarize_power.py`; it verifies capture hashes and recomputes energy before plotting. Model predictions are stored in a separate directory and never replace measured samples.
