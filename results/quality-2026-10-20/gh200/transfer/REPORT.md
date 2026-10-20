# External GPU power-model evaluation

Fixed model fitted only on training GPU; test GPU used solely for evaluation. No clipping or adaptation. Correlated telemetry rows are not independent trials. Different capture dates and environments confound attribution to hardware alone.

Training rows: 1123. Test rows: 1237. Fixed model: `linear_regression`.

| Model | MAE (W) | RMSE (W) | R² |
| --- | ---: | ---: | ---: |
| linear_regression | 23.003 | 30.838 | 0.2116 |
| training_mean | 38.926 | 48.958 | -0.9872 |
| training_workload_mean | 34.737 | 39.594 | -0.2997 |

Baselines use training targets only. Unseen workloads fall back to the training global mean.
The model is saved and reloaded before scoring; transfer scores are kept separate from model metadata.

[Evidence](evaluation.json) · [Predictions](predictions.csv) · [Per-workload errors](per_workload.csv)
