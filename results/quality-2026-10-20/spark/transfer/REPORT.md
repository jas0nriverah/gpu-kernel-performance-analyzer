# External GPU power-model evaluation

Fixed model fitted only on training GPU; test GPU used solely for evaluation. No clipping or adaptation. Correlated telemetry rows are not independent trials. Different capture dates and environments confound attribution to hardware alone.

Training rows: 1123. Test rows: 1023. Fixed model: `linear_regression`.

| Model | MAE (W) | RMSE (W) | R² |
| --- | ---: | ---: | ---: |
| linear_regression | 222.480 | 250.111 | -74.4646 |
| training_mean | 393.770 | 394.821 | -187.0531 |
| training_workload_mean | 392.855 | 394.890 | -187.1183 |

Baselines use training targets only. Unseen workloads fall back to the training global mean.
The model is saved and reloaded before scoring; transfer scores are kept separate from model metadata.

[Evidence](evaluation.json) · [Predictions](predictions.csv) · [Per-workload errors](per_workload.csv)
