# Usage

Build and benchmark commands are in the [README](../README.md). Full GPU validation and Nsight import are in [reproducibility](reproducibility.md).

## CPU fixture

Exercises the CLI without CUDA. Outputs are synthetic, not GPU measurements.

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

## Resume a sweep

Repeat an interrupted sweep with `--resume`. The checkpoint validates the binary, configuration, device UUID, and completed results before reuse. Partial and final directories must be on the same filesystem. POSIX file locks prevent concurrent writers; recovery covers process interruption, not sudden power loss.

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/vector-add

python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir outputs/vector-add \
  --resume
```

## Compare runs

Use matching GPU and warmup/repeat settings. The default statistic is median runtime. With `--fail-on-regression`, regressions, noise, missing coverage, or protocol changes return 1; invalid inputs return 2. Cross-device comparisons require `--allow-device-mismatch` and cannot pass a CI gate.

```bash
python -m gpu_kernel_analyzer compare \
  --baseline outputs/before --candidate outputs/after \
  --outdir outputs/comparison --threshold-pct 5 --max-cv 0.10 \
  --fail-on-regression
```

## Tune block sizes

The tuner checks correctness, then measures runtime and power. It selects minimum estimated energy among candidates within 5% of the fastest, and independently repeats eligible candidates. Incomplete confirmation, unstable results, or indistinguishable energy minima return nonzero. This dispersion check is not a significance test. Use an idle GPU and a fresh output directory.

```bash
gpu-kernel-analyzer autotune run \
  --binary build/gpu_benchmark \
  --kernels vector_add \
  --problem-sizes 1048576 \
  --block-sizes 128 256 512 1024 \
  --outdir outputs/vector-add-autotune \
  --device 0
```

## Capture power

Power capture uses sustained workloads and sampled GPU telemetry. See [measurement scope](methodology.md#sustained-power-and-energy); estimated energy is not an isolated per-kernel measurement.

```bash
python -m pip install -e '.[dev,power,api,lint]'
cmake -S benchmarks -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
python -m gpu_kernel_analyzer power capture \
  --binary build/gpu_benchmark --scenarios configs/power_h100.yaml \
  --device 0 --duration-seconds 15 --trials 3 \
  --outdir outputs/h100-power
python scripts/summarize_power.py --run-dir outputs/h100-power
```

## Train a power model

```bash
gpu-power-pipeline run --source local_nvidia_smi \
  --data-path outputs/h100-power/steady_telemetry.csv \
  --split-strategy grouped --run-validation --generate-report \
  --save-model --registry-dir outputs/power-registry \
  --outdir outputs/h100-power-model
```

Grouped validation holds out complete trials, not adjacent telemetry rows. For the published frozen model, use [reproduction](reproducibility.md).

## Exploratory runtime model

The ridge regression model predicts runtime and effective bandwidth from kernel name, problem size, and block size. It uses scikit-learn with the `[ml]` extra or a NumPy fallback. This is an exploratory extension to the measured benchmarking pipeline.

```bash
python -m gpu_kernel_analyzer model train \
  --run-dir outputs/run --model-out outputs/perf_model.json

python -m gpu_kernel_analyzer model predict \
  --model outputs/perf_model.json --kernel gemm_tiled --problem-size 512 --block-size 16

python -m gpu_kernel_analyzer model recommend-block-size \
  --model outputs/perf_model.json --kernel vector_add --problem-size 4194304 \
  --candidates 64,128,256,512

python -m gpu_kernel_analyzer advise --run-dir outputs/run
```

Predictions are tagged `predicted` and written separately from measured artifacts. Training supports repeated `--run-dir` arguments, but **do not pool different GPUs**: the current model has no device feature. Small datasets and unseen block sizes trigger warnings. Leave-one-out scores on nearby configurations do not establish generalization to a new workload or GPU; test on held-out sizes and independent runs before trusting recommendations.

The advisor is deterministic and offline. Its optional `--llm` switch is a stub, not a bundled AI integration. No API key is needed for ordinary use.
