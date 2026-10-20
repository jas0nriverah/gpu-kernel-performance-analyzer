# Reproducibility

## Research figures, model and SQL on CPU

The [research report](research-report.md) describes the hypotheses, split policy, frozen selection and limits. Use Python 3.11 for the recorded scientific dependencies; the general package supports Python 3.10+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[power,api]" -c requirements-research.txt
python scripts/plot_research_results.py
python scripts/reproduce_frozen_model.py --outdir outputs/model-replay
```

The figure generator reads checked-in CSV/JSON only, verifies key counts and scores, and exports seven PNG/SVG pairs plus calculation CSVs. [The gallery manifest](../results/research-2026-10-20/figure-manifest.json) records each input/output SHA-256 and the plotting environment. Rendering bytes may vary with library/font versions; scientific calculations and input identity are checked separately.

Model replay verifies the archived plan, selection, training-source and training-input hashes, relocates the two development inputs by content, and reconstructs the exact selected configuration. It saves a loadable registry, verifies save/reload parity, and checks every archived final prediction and metric for H200, GH200 and GB10 within 0.000001 W numerical tolerance. The recorded environment reproduced the original model's serialized SHA-256 and all 3,413 final predictions to within 6 × 10⁻¹⁴ W. A different software environment may fail this strict comparison; inspect that difference rather than altering the historical results.

This is **retrospective numerical reproduction**, not a newly collected test. The replay does not rewrite historical freeze timestamps or claim the old captures happened after the new fit. For a new experiment, freeze a new selection and collect another independent capture; the strict prospective evaluator rejects earlier captures. No downloaded pickle is needed: weights are rebuilt from the checked-in measured data. Use a new or empty output directory.

To serve the reconstructed model:

```bash
GPU_POWER_REGISTRY_DIR=outputs/model-replay/registry \
  GPU_POWER_MODEL_NAME=rf_unbounded_full GPU_POWER_MODEL_VERSION=latest \
  python -m uvicorn gpu_power_pipeline.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API and `/model` for required inputs and accepted GPU names. See the [serving contract](serving.md).

Rebuild the complete measured power warehouse without CUDA:

```bash
python -m gpu_power_pipeline.warehouse ingest --database outputs/research-power.sqlite \
  results/h100-power-2026-09-22 \
  results/quality-2026-10-20/h200/power \
  results/quality-2026-10-20/gh200/power \
  results/quality-2026-10-20/h200-independent/power \
  results/quality-2026-10-20/spark/power
python -m gpu_power_pipeline.warehouse query --database outputs/research-power.sqlite
```

Expected totals: 5 captures, 120 workload windows, 10,301 raw samples and 5,689 steady rows. Repeating ingestion adds no duplicates. Verify the curated evidence using `sha256sum -c SHA256SUMS` from `results/quality-2026-10-20/`. The original manifests retain capture-time paths; the archive README explains relocation and omitted executables. Scientific dependency constraints are in `requirements-research.txt`; full capture environments remain in the evidence archive.

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
