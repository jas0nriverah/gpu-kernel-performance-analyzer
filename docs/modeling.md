# Predictive models and reproducible evaluation

The repository contains two distinct predictive tasks. The measured-power study predicts contemporaneous GPU sensor watts from steady telemetry; the optional performance model predicts runtime/bandwidth from scenario configuration. Their targets, splits and artifacts must not be combined. [Research report](research-report.md) · [Model card](model-card.md) · [Measurement methodology](methodology.md)

## Measured GPU power

The selected known-device model is a random forest with 64 trees, unrestricted depth and minimum leaf size 3 (seed 42; one worker). It consumes GPU/memory utilization, graphics/memory clocks, temperature, workload and GPU name. Numeric median imputation plus standardization and categorical mode imputation plus one-hot encoding fit inside each training fold. The model predicts one steady telemetry row at a time; SQL trial aggregates are analytical summaries, not its training target.

![Actual preprocessing and selected ensemble architecture](assets/ml-model.svg)

The tree sketches are conceptual, not explanations extracted from fitted trees. No feature-importance ranking is inferred from unavailable weights.

Twelve fixed candidates cover linear regression, two random forests and two gradient-boosting configurations under full/portable feature sets, plus global/workload-mean baselines. Explicit feature allowlists exclude identities, timestamps and target-derived quantities. See [the frozen grid and chronology](model-generalization.md).

For known-device sessions, six whole-device/trial validation folds selected the forest at **3.108 W** mean RMSE. After selection froze, one new H200 capture produced **9.556 W** final RMSE versus **23.157 W** for the development-workload baseline. All residuals remain, including 24 errors above 25 W. The difference between validation and final error is visible and intentional.

For the distinct unseen-device objective, two directional H100/H200 folds selected a workload baseline. Final GH200 and GB10 tests expose failed portability; the known-device forest scored **42.990 W** and **379.277 W** respectively. No final-test-driven winner change occurred. These models are reference-device limited, not general GPU power predictors.

## Reproduce a frozen model on CPU

The [curated archive](../results/quality-2026-10-20/README.md) includes training captures, frozen plan/selection metadata and final predictions, but deliberately omits pickled weights. Reconstruct the recorded selected model into a new directory:

```bash
python -m pip install -e '.[power,api]'
python scripts/reproduce_frozen_model.py --outdir outputs/model-replay
```

The replay checks archived provenance and compares reconstructed predictions/metrics with archived final evidence. It is retrospective reproduction, not a new prospective experiment; timestamps are not backdated. Run it with the recorded dependency versions when checking numerical parity. See [reproducibility](reproducibility.md) for the service startup path and [serving contract](serving.md) for accepted feature types and readiness.

To repeat development from checked-in inputs, use a new output directory:

```bash
python -m gpu_power_pipeline.generalization \
  --h100 results/h100-power-2026-09-22/steady_telemetry.csv \
  --h200 results/quality-2026-10-20/h200/power/steady_telemetry.csv \
  --outdir outputs/model-development-replay
python scripts/fit_known_device_model.py \
  --development-dir outputs/model-development-replay \
  --outdir outputs/known-device-replay
```

A new final-test claim requires a new untouched capture after its own freeze. `final_evaluation --scope new-capture` correctly rejects historical sessions preceding a newly reconstructed selection. Preserve the original selection and final evidence instead of relabeling that replay as prospective.

## Offline evaluation versus serving

Offline evaluation tolerates unseen categories and missing optional sensor values using the training-fitted transformations. The API additionally rejects GPU names outside the loaded reference profile before prediction, validates required/finite physical values and bounds batches. The selected forest served every prospective H200 row with exact parity; both frozen selected bundles rejected all 1,023 GB10 rows. This guard is operational protection, not evidence that the offline model transfers accurately. Historical H100-only and later pooled H100/H200 bundles have different reference identities.

## Exploratory runtime prediction and tuning

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
