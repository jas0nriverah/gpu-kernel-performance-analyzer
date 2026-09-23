# H100 power-model validation

Models were trained on **1,123 real H100 telemetry samples** from eight configurations across three randomized trials. Each cross-validation fold holds out one entire trial. The same workloads and GPU occur in all trials; this evaluates repeat-session generalization only.

| Model | Mean held-out MAE (W) | Mean held-out RMSE (W) | Mean held-out R² |
| --- | ---: | ---: | ---: |
| Linear regression | 2.38 | 3.11 | 0.9873 |
| Random forest | 2.38 | 3.16 | 0.9868 |

These errors describe prediction of sampled board-power readings, not calibration against an external power meter. Neighboring readings are correlated. The limited feature/workload range and a single GPU restrict generalization. No cross-GPU, unseen-workload, or deployment accuracy claim follows from these numbers.

[Every fold](validation/grouped_blocked_cv.csv) · [CV summary](validation/grouped_blocked_cv_summary.csv) · [Dataset checks](data/DATASET_REPORT.md) · [Measured input telemetry](../h100-power-2026-09-22/steady_telemetry.csv) · [Source and input hashes](EVIDENCE.json)

The saved model uses linear regression, selected by the single grouped holdout. The model was then fitted on all available rows for serving. This is exploratory model selection, not nested cross-validation. Its [metadata](saved_model_metadata.json) lists the exact input features. Run/session IDs, source filenames, GPU UUIDs, timestamps, and trial/scenario identifiers are excluded from model features.

The fitted model was reloaded successfully. An in-process FastAPI health/prediction check passed, and its prediction matched direct batch inference. [API check](api_validation.json). That check verifies serving behavior, not out-of-sample prediction accuracy. The serialized model remains in the local `outputs/merge-review/h100-power-registry` directory; it is reproducible with the command below rather than committed as a pickle.

## Reproduce

```bash
python -m pip install -e '.[power,api]'
python -m gpu_power_pipeline run --source local_nvidia_smi \
  --data-path results/h100-power-2026-09-22/steady_telemetry.csv \
  --split-strategy grouped --run-validation --generate-report \
  --save-model --registry-dir outputs/h100-power-registry \
  --outdir outputs/h100-power-model
```

The [dependency lock](../h100-power-2026-09-22/requirements-lock.txt) records the validation environment. Source revisions, inputs, and model-bundle hashes are recorded in `EVIDENCE.json`; the run was performed from an uncommitted integration branch. The full generated diagnostics remain in [ONE_PAGE_REPORT.md](ONE_PAGE_REPORT.md). Synthetic CLI smoke results are separate and are not used in the table above.
