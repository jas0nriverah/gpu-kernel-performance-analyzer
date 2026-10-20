# External GPU power-model evaluation

Fixed model fitted only on training GPU; test GPU used solely for evaluation. No clipping or adaptation. Correlated telemetry rows are not independent trials. Different capture dates and environments confound attribution to hardware alone.

Training rows: 1123. Test rows: 1153. Fixed model: `linear_regression`.

| Model | MAE (W) | RMSE (W) | R² |
| --- | ---: | ---: | ---: |
| linear_regression | 50.507 | 56.779 | -0.7913 |
| training_mean | 38.660 | 43.000 | -0.0274 |
| training_workload_mean | 26.337 | 29.487 | 0.5169 |

Baselines use training targets only. Unseen workloads fall back to the training global mean.
The model is saved and reloaded before scoring; transfer scores are kept separate from model metadata.

[Evidence](evaluation.json) · [Predictions](predictions.csv) · [Per-workload errors](per_workload.csv)
