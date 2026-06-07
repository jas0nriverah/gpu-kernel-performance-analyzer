#GPU Kernel Performance Analyzer

Benchmarks CUDA kernels with CUDA-event timing, derives bandwidth, GFLOPs, and arithmetic intensity, classifies kernels as memory- or compute-bound, and optionally overlays Nsight Compute profiler metrics.

The project also includes a performance-prediction model and tuning advisor built on measured benchmark data. It can predict runtime for unseen configurations, rank candidate block sizes, and generate plain-language tuning recommendations.

Architecture

flowchart LR
    A[CUDA Kernels] --> B[Benchmark Harness]
    B --> C[JSON/CSV Artifacts]
    C --> D[Analysis Pipeline]
    C --> E[Performance Model]
    C --> F[Tuning Advisor]
    D --> G[Reports]
    D --> H[Plots]
    D --> I[Roofline Analysis]
    E --> J[Runtime Prediction]
    E --> K[Block-size Ranking]
    F --> L[Optimization Recommendations]
    M[Nsight Compute CSV] -. optional import .-> C
    subgraph CUDA Layer
        A
        B
    end
    subgraph Artifact Layer
        C
        M
    end
    subgraph Python Layer
        D
        E
        F
    end
    subgraph Outputs
        G
        H
        I
        J
        K
        L
    end

What this demonstrates

* CUDA/C++ kernels with a timing harness, plus Python orchestration, analysis, plotting, and reporting.
* Metric provenance: each metric is tagged as measured, derived_estimate, predicted, or unavailable.
* Nsight Compute integration with validation that only imported profiler metrics are marked as measured.
* ML on the project’s own benchmark data: runtime and bandwidth prediction, block-size autotuning, leave-one-out cross-validation, and small-data warnings.
* CPU-only testing support through committed sample artifacts and a fake benchmark binary.

Supported kernels

Kernel	Computes	Behavior it shows
vector_add	c[i] = a[i] + b[i]	Classic memory-bound kernel; effective bandwidth is the headline metric.
memcpy_bandwidth	out[i] = in[i]	Pure streaming copy; closest single-kernel proxy for achievable DRAM bandwidth.
stencil_1d	3-point weighted stencil	Modest data reuse and boundary handling; between copy and GEMM, still memory-bound.
reduction	block-wise tree sum	Shared memory, __syncthreads(), and the tree-reduction pattern.
gemm_naive	dense C = A*B	Compute-heavy but un-tiled; re-reads global memory.
gemm_tiled	dense C = A*B, 16x16 tiles	Shared-memory tiling that cuts DRAM traffic; optimized baseline for the speedup comparison.

Metrics and provenance

Each value records how it was produced, so measured data, derived metrics, profiler counters, and model predictions remain separate.

Metric	Status	Source
runtime_ms	measured	CUDA events (cudaEventElapsedTime)
effective_bandwidth_GBps	derived_estimate	bytes_moved / runtime
effective_GFLOPs	derived_estimate	flops / runtime
arithmetic_intensity	derived_estimate	flops / bytes_moved
device_metadata	measured	CUDA runtime API
occupancy, sm_utilization, memory_throughput_pct, l2_throughput_pct, l2_cache_hit_rate	unavailable until imported, then measured	Nsight Compute CSV import
model predictions	predicted	Performance model, written to model_predictions.csv

CUDA-event timing is the source of truth for runtime_ms. Nsight Compute is used for hardware counters, not benchmark timing. Predicted values are written to separate artifacts and are never inserted into measured benchmark files.

Quickstart

1. Build the CUDA harness

Requires an NVIDIA GPU and nvcc.

cmake -S benchmarks -B build
cmake --build build --config Release

Binary:

build/gpu_benchmark

On Windows:

build/Release/gpu_benchmark.exe

Set CUDAToolkit_ROOT if CMake cannot find the CUDA Toolkit.

2. Run a benchmark sweep

python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/run

3. Validate artifacts

python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/run

4. Analyze results

python -m gpu_kernel_analyzer analyze full --run-dir outputs/run \
  --peak-gflops <real_fp32_peak> --peak-bandwidth-gbps <real_dram_peak>

--peak-* values are optional. If omitted, the roofline plot shows only measured operating points. Supplying real GPU peak FP32 throughput and DRAM bandwidth adds roofline ceilings.

CPU-only fixture demo

The Python pipeline can run without a GPU using a fake benchmark binary. Fixture outputs are sample-only and should not be cited as real GPU performance.

python -m gpu_kernel_analyzer benchmark sweep \
  --binary tests/fixtures/fake_benchmark.py --binary-interpreter python \
  --scenarios configs/benchmark_scenarios.yaml --outdir outputs/sample_fixture
python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/sample_fixture
python -m gpu_kernel_analyzer analyze full --run-dir outputs/sample_fixture

Performance model and autotuning

The model predicts runtime_ms and effective_bandwidth_GBps from kernel name, problem size, and block size. It uses ridge regression in log space, with scikit-learn when the optional [ml] extra is installed and a NumPy fallback otherwise.

# Train from one or more run directories
python -m gpu_kernel_analyzer model train \
  --run-dir outputs/run --model-out perf_model.json
# Predict runtime and bandwidth for a configuration without running it
python -m gpu_kernel_analyzer model predict \
  --model perf_model.json --kernel gemm_tiled --problem-size 512 --block-size 16
# Rank candidate block sizes by predicted runtime
python -m gpu_kernel_analyzer model recommend-block-size \
  --model perf_model.json --kernel vector_add --problem-size 4194304 \
  --candidates 64,128,256,512
# Generate rule-based tuning recommendations from measured artifacts
python -m gpu_kernel_analyzer advise --run-dir outputs/run

Model behavior and limitations:

* Predictions are tagged predicted and written to model_predictions.csv.
* Training accepts multiple --run-dir flags and pools the results.
* With limited data, training prints a warning that cross-validation metrics are not statistically meaningful.
* Block-size recommendations are flagged as extrapolations when the training data does not contain varied block sizes for that kernel.
* The advise command is deterministic and offline. The optional --llm flag is a stub; no LLM dependency or API key is required for normal use or tests.

Sample output

analyze full writes a self-contained run directory:

outputs/run/
├── run_manifest.json            # git commit, device, environment, scenario hashes, Nsight status
├── timing_samples.csv           # every raw CUDA-event sample
├── benchmark_summary.csv        # per-scenario statistics and derived metrics
├── metrics_provenance.csv       # metric status: measured, derived, predicted, or unavailable
├── analysis_heuristics.csv      # memory- vs compute-bound classification
├── analysis_speedup.csv         # tiled vs naive GEMM speedup
├── plots/
│   ├── runtime_vs_size_gemm.png
│   ├── effective_gflops_vs_size_gemm.png
│   ├── effective_bandwidth_vs_size_vector_reduction.png
│   ├── effective_bandwidth_vs_size_memory_kernels.png
│   └── roofline.png
└── REPORT.md                    # human-readable summary of the run

Real GPU validation snapshot

Validation run: NVIDIA A100 80GB PCIe.

The numbers below were captured with an earlier scenario set. The newer memcpy_bandwidth and stencil_1d kernels have not yet been re-validated on a GPU.

* GPU: NVIDIA A100 80GB PCIe
* CUDA Toolkit: 13.0
* Python: 3.11.9
* Artifact validation: passed
* Nsight Compute: imported for selected scenarios only

Comparison, 512x512 GEMM	Effective GFLOPs
gemm_tiled	~3840
gemm_naive	~2589
Speedup, tiled vs naive	~1.48x

Imported Nsight Compute metrics for vector_add, size 4194304, block size 256:

Metric	Value
Occupancy	~76.72%
SM utilization	~21.36%
Memory throughput	~71.42%
L2 throughput	~77.37%
L2 cache hit rate	unavailable

A sanitized copy of this capture is committed at:

examples/ncu_raw_vector_add_4194304_b256.csv

Optional Nsight Compute workflow

# Detect Nsight Compute
python -m gpu_kernel_analyzer profile ncu-detect
# Print raw-capture and import commands for one scenario
python -m gpu_kernel_analyzer profile ncu-plan --binary build/gpu_benchmark \
  --kernel vector_add --problem-size 4194304 --block-size 256 --warmups 10 --repeats 30 --verify \
  --raw-csv-out outputs/run/ncu_raw_vector_add.csv \
  --normalized-csv outputs/run/ncu_norm_vector_add.csv --run-dir outputs/run
# Normalize a raw Nsight CSV into import format
python -m gpu_kernel_analyzer profile ncu-normalize \
  --raw-csv examples/ncu_raw_vector_add_4194304_b256.csv \
  --out-csv outputs/run/ncu_norm_vector_add.csv \
  --kernel vector_add --problem-size 4194304 --block-size 256 --metric-set default_profiler_set
# Import normalized metrics
python -m gpu_kernel_analyzer profile ncu-import --run-dir outputs/run \
  --source-tool ncu --source-file ncu_raw_vector_add.csv \
  --metric-set default_profiler_set --ncu-csv outputs/run/ncu_norm_vector_add.csv

Profiler metrics are rejected unless they map to an exact benchmark scenario.

Project structure

benchmarks/                 CUDA/C++ harness
├── include/                kernel and harness headers
└── src/
    ├── benchmark_runner.cu CLI parsing and JSON output
    ├── cuda_utils.cu       device query and CUDA error checks
    └── kernels/            one .cu file per kernel
configs/
├── benchmark_scenarios.{yaml,json}  sweep definitions
└── ncu_metric_sets.yaml             Nsight metric-set mappings
src/gpu_kernel_analyzer/    Python package
├── cli.py                  command-line interface
├── scenarios.py            config loading, expansion, validation, and warnings
├── runner.py               benchmark binary invocation
├── metrics.py              summary statistics, derived metrics, and provenance
├── analysis.py             bottleneck heuristics and speedup analysis
├── plotting.py             matplotlib plots and roofline analysis
├── report.py               Markdown report generation
├── perf_model.py           performance prediction and autotuning
├── advisor.py              rule-based tuning recommendations
├── ncu.py                  Nsight capture plan, normalization, and import
├── artifacts.py            artifact schema and validation
└── ...
examples/                   sanitized sample Nsight raw CSV
tests/                      pytest suite and CPU-only fixtures
docs/                       methodology, metrics policy, and reproducibility notes

Design notes

* CUDA-event timing: cudaEvent timestamps are recorded on the GPU stream, measuring device-side kernel execution without host-launch or CPU-clock noise.
* Warmups and repeated trials: warmups absorb one-time costs such as context setup, JIT compilation, allocation, cold caches, and clock ramp-up. Repeated trials produce min, median, mean, p95, standard deviation, and coefficient of variation.
* Profiler overhead: Nsight Compute can serialize and replay kernels, so its timing is not used for runtime_ms. Nsight is used for hardware counters only.
* Tiled vs naive GEMM: naive GEMM re-reads A and B from global memory for every output element. Tiled GEMM stages 16x16 tiles in shared memory, reuses each loaded value across a tile, reduces DRAM traffic, and raises arithmetic intensity.
* Derived metrics: each kernel declares bytes moved and FLOPs performed. The analyzer computes bandwidth, GFLOPs, and arithmetic intensity from those declarations and the measured runtime.

Validating on a real GPU

On a machine with an NVIDIA GPU and the CUDA Toolkit, this script builds the harness, smoke-tests the kernels, runs the benchmark sweep, validates artifacts, and runs the test suite:

pip install -e ".[dev]"
bash scripts/validate_gpu.sh

Optional roofline ceilings:

PEAK_GFLOPS=19500 PEAK_BW_GBPS=1935 bash scripts/validate_gpu.sh

The script honors BUILD_DIR, OUT_DIR, PYTHON, and SKIP_PIP_INSTALL overrides. Set CUDAToolkit_ROOT if CMake cannot locate the CUDA Toolkit.

Testing and CI

pip install -e ".[dev]"
python -m pytest -q

The CPU-only test suite covers:

* config parsing and validation
* summary statistics
* derived metrics
* speedup analysis
* artifact-schema validation
* report and plot generation
* roofline analysis
* Nsight normalization and import
* performance modeling
* autotuning
* predicted-provenance separation
* tuning recommendations

CI runs on CPU-only GitHub Actions across Python 3.10 to 3.12, plus a ruff lint job. It exercises the Python pipeline using committed sample artifacts and a fake benchmark binary. CUDA benchmarks require a CUDA-capable GPU and nvcc locally.