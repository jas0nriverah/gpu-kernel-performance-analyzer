# Model card: known-device GPU power prediction

The selected learned model is a **64-tree random forest** for a later capture on the known H100/H200 devices. A separate experiment asks whether a model transfers to an unseen GPU; that objective selected a workload-mean baseline. These are different policies and claims.

![Selected model architecture](assets/ml-model.svg)

The tree sketches illustrate ensemble averaging. They are not fitted trees, feature attributions, or explanations of a particular prediction.

## Target and intended use

Each example is one trimmed, steady telemetry sample. The target is `power_watts` from `nvidia-smi:power.draw`, aligned with utilization, clocks, temperature, workload and device name. Models load the captured **steady CSV rows**, not window-aggregate or SQL-aggregate rows. Window summaries support measurement reporting and integrity checks; SQL provides a validated evidence mirror.

The supported serving scope is the approved H100/H200 GPU profile under the measured workloads. This is an exploratory estimator, not a power controller or a full-server energy model. On integrated GB10, NVIDIA identifies this sensor as GPU-only, excluding CPU and full SoC/system power; the historical `gpu_board` schema label does not expand that measurement scope. See the [Spark sensor clarification](dgx-spark-validation.md).

## Inputs and preprocessing

| Input | Type / unit |
| --- | --- |
| `gpu_utilization_pct` | Numeric, percent |
| `memory_utilization_pct` | Numeric, percent; memory activity, not memory capacity |
| `graphics_clock_mhz` | Numeric, MHz |
| `memory_clock_mhz` | Numeric, MHz |
| `temperature_c` | Numeric, °C |
| `workload_type` | Categorical; six measured kernel types |
| `gpu_name` | Categorical; approved H100/H200 names |

GPU UUID, timestamp, run/session/scenario/trial identifiers and the target are excluded from predictive inputs. A scikit-learn `ColumnTransformer` applies median imputation and `StandardScaler` to numeric columns, and most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")` to categorical columns. These shared preprocessing steps are fitted **only on training rows inside each validation fold**, then refitted with the selected estimator on all development rows. Scaling is part of the actual shared pipeline, not a claim that trees require standardized inputs.

Offline preprocessing can impute missing numeric sensors. The live API has a stricter contract: required fields, finite values, physical ranges and an approved `gpu_name` must pass validation before prediction. Unknown-category tolerance in the encoder does not bypass that device guard. `/model` advertises `required_fields` and `allowed_gpu_names`; approval is independent of whether a feature set uses GPU name for prediction.

## Estimator and frozen selection

| Setting | Saved known-device model |
| --- | --- |
| Candidate / estimator | `rf_unbounded_full` / `RandomForestRegressor` |
| Trees / bootstrap | `64` / `True` |
| Split criterion / features per split | `squared_error` / `max_features=1.0` |
| Depth / minimum leaf rows | `max_depth=None` / `min_samples_leaf=3` |
| Random state / estimator threads | `42` / `n_jobs=1` |
| Prediction | Arithmetic mean of the 64 tree predictions, in watts |
| Saved version | `20261020T204029Z` |

Each tree learns a sequence of splits in the transformed telemetry. A prediction reaches a leaf that returns the mean training target for that leaf. Bootstrap resampling gives the trees different training samples; averaging their outputs reduces dependence on one fitted tree. This mechanism does not establish accuracy on a new device.

Development used 2,276 steady rows from the original H100 and H200 captures. The predefined 12-candidate plan compared full and portable feature sets, learned estimators, and two training-only mean baselines. Known-device selection minimized the unweighted mean RMSE across six whole GPU/trial holdouts. Its mean validation RMSE was **3.108 W**; this is a development score. Selection froze at `2026-10-20T20:40:29Z`, before the later H200 capture was collected. No uncertainty or probability calibration stage was performed.

The separate unseen-device policy selected `training_workload_mean` using two whole-device development holdouts. It predicts the training power mean for a workload, falling back to the global training mean for an unseen workload. It is a baseline, not a tree ensemble. Both policies were frozen before external evaluation. See the [known-device selection](../results/quality-2026-10-20/models/known-device-development/selection.json), [unseen-device selection](../results/quality-2026-10-20/models/model-development/selection.json), and [research workflow](assets/research-workflow.svg).

## Independent results and failed transfer

| Frozen policy / test | Steady rows | MAE, W | RMSE, W | R² |
| --- | ---: | ---: | ---: | ---: |
| RF / later H200 capture | 1,153 | 2.319 | 9.556 | 0.949 |
| Training workload mean / later H200 | 1,153 | 18.740 | 23.157 | 0.699 |
| RF / GH200 stress test | 1,237 | 37.007 | 42.990 | −0.532 |
| Frozen unseen-device workload mean / GH200 | 1,237 | 31.089 | 34.714 | 0.001 |
| RF / GB10 stress test | 1,023 | 378.728 | 379.277 | −172.537 |
| Frozen unseen-device workload mean / GB10 | 1,023 | 388.292 | 391.267 | −183.682 |

The later H200 test supports the known-device, new-session claim for this capture. Its RMSE is substantially higher than its MAE and development RMSE; inspect residual and workload plots rather than relying on one headline score. GH200 and GB10 tests did **not** establish portability. The known-device RF was evaluated there as an offline stress test, not as an approved serving deployment. Negative R² means worse squared error than that test set's constant mean reference; the test mean was not used to train or select a model.

Frozen evaluation evidence: [later H200](../results/quality-2026-10-20/h200-independent/evaluation/evaluation.json), [GH200 RF](../results/quality-2026-10-20/gh200/known-device-challenge/evaluation.json), [GH200 selected baseline](../results/quality-2026-10-20/gh200/frozen-evaluation/evaluation.json), [GB10 RF](../results/quality-2026-10-20/spark/known-device-challenge/evaluation.json), [GB10 selected baseline](../results/quality-2026-10-20/spark/frozen-evaluation/evaluation.json).

## Limits and deployment checks

Samples within a sustained window are correlated; trial ranges and row counts do not provide independent confidence intervals. Coverage is six custom FP32/streaming kernels, a small set of device identities, and limited sessions. Driver/toolkit, host CPU, temperature and capture-date differences confound device comparisons. There is no fitted-tree explanation, causal interpretation, uncertainty interval, or general energy-efficiency guarantee.

The service checks model identity, finite output and batch limits; prediction logs and drift/service counters support inspection. Offline drift warnings are diagnostics, not an automatic production safety mechanism or a demonstrated explanation of errors. Preserve the frozen bundle, preprocessing, software versions and approved-device profile together. Model metadata and hashes are published in the evidence archive; joblib weights are deliberately omitted there. See [serving](serving.md), [modeling source](../src/gpu_power_pipeline/generalization.py), [preprocessing source](../src/gpu_power_pipeline/preprocessing.py), and [evidence reproduction](../results/quality-2026-10-20/README.md).
