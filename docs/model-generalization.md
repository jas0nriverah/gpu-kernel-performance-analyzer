# Device-held-out power-model development

This experiment used H100 and H200 captures for development and reserved GH200 for final evaluation. The [research report](research-report.md), [model card](model-card.md) and [curated evidence archive](../results/quality-2026-10-20/README.md) present the completed study. The earlier fixed H100→H200 result remains a historical external-device test. Once H200 outcomes guide feature or model selection, H200 is development data and cannot provide an untouched final-test claim.

## Frozen selection protocol

Use only the measured steady-window rows from the H100 and H200 captures. Fit preprocessing and weights independently inside each development fold: H100→H200 and H200→H100. Select the candidate with the lowest arithmetic mean of the two fold RMSEs, giving each held-out device equal weight. Report both directional scores and training-only global/workload-mean baselines. A simple baseline may win; model complexity does not establish quality.

The bounded candidate plan compares linear regression, two fixed random forests and two fixed gradient-boosting configurations, each with two explicit feature sets. Global-mean and workload-mean estimators also participate in selection. Preprocessing uses training-fold imputation, scaling and one-hot encoding with unknown categories ignored. No sample-wise random split is used for selection.

The full feature set includes GPU/memory utilization, graphics/memory clocks, temperature, workload and GPU name. The portable ablation removes GPU name and memory clock. In these two development captures, GPU utilization is always 100%, graphics clock is always 1980 MHz, and memory clock is constant within each device (2619 MHz H100; 3201 MHz H200). Thus a static clock or GPU-name feature cannot establish a calibrated response to unseen hardware from either single-device training fold. Memory utilization, temperature and workload contain the observed within-device variation.

Explicit feature allowlists exclude targets and all row identities, timestamps, elapsed/query durations, session/run/trial/scenario IDs, UUIDs, source paths and capture-phase artifacts. Labels, target-derived summaries and energy estimates never become predictors. Workload is a valid input here, so this experiment does not test unseen workload-family generalization.

Whole-trial grouped diagnostics are reported separately. They test repeat-session behavior on the two development GPUs and do not influence cross-device candidate selection. Nearby telemetry rows are correlated; a row count is not a count of independent experiments. Three trials per capture and two development devices do not support broad population claims or precise confidence intervals.

## Final evaluation boundary

Write an immutable selection artifact containing the candidate grid, feature contract, two-direction development scores, chosen configuration and development input/source hashes before accessing GH200 labels. Fit only that chosen configuration on the combined H100/H200 development rows, save it, reload it and check prediction parity. Evaluate the resulting bundle on GH200 once, against global/workload baselines derived only from development targets. Keep predictions and errors by workload, device and trial where available.

GH200 scores must not trigger a new grid, feature revision, clipping rule or calibration while preserving a final-test claim. Such revisions require another independent untouched capture. Cross-device results also differ in host, capture date, CUDA/build and environment; accuracy differences cannot isolate hardware alone. Board-power predictions describe this telemetry protocol, not isolated kernel or whole-server energy.

## Evidence status

The frozen two-device search selected the training-workload-mean baseline at **25.4766 W mean directional RMSE**. The best learned candidate, portable gradient boosting (64 trees, depth 2, learning rate 0.05), scored **33.7206 W**. This is a development result: the learned candidates did not beat the simple baseline for the chosen cross-device objective. The [archived development artifacts](../results/quality-2026-10-20/models/model-development/selection.json) retain the original local `outputs/` paths as provenance; no GH200 score was accessed for this selection.

A separate operational objective asks about **future sessions on known H100/H200 GPUs**. It uses the same fixed twelve candidates and the six already-recorded whole-device/trial diagnostic folds. Other trials on the held-out trial's GPU remain in training, so these scores cannot support unseen-GPU claims. The resulting model is independently frozen for a prospective new capture; it does not replace the device-held-out winner.

| Known-device candidate | Mean six-fold validation RMSE (W) | Worst fold RMSE (W) |
| --- | ---: | ---: |
| Random forest, full features, unrestricted depth | 3.1080 | 5.5052 |
| Random forest, portable features, unrestricted depth | 3.4187 | 5.7924 |
| Random forest, full features, depth 6 | 5.4782 | 11.8226 |
| Training-workload mean | 19.1869 | 24.9355 |
| Training global mean | 35.3499 | 42.8399 |

The selected forest has 64 trees and minimum leaf size 3, with seed 42 and one worker. These are selection/validation scores, not independent test scores. `scripts/fit_known_device_model.py` checks the original plan, selection, implementation and input hashes, requires each candidate's six unique folds and expected row counts, and writes exclusive plan/selection files before fitting. Its known-device selection froze at **2026-10-20 20:40:29.710867 UTC**, before collection of the prospective H200 validation capture. The saved bundle fitted 2,276 H100/H200 rows and passed saved/reloaded prediction parity. Final evaluation in `new-capture` mode requires an explicit valid UTC selection-freeze timestamp, every test timestamp strictly after that freeze and all development rows, disjoint session/run IDs and a previously trained GPU UUID.

Reconstruct the archived forest from checked-in training captures without the original local paths:

```bash
python scripts/reproduce_frozen_model.py --outdir outputs/model-replay
```

This is retrospective numerical reproduction, not a new freeze or prospective test. The [known-device selection](../results/quality-2026-10-20/models/known-device-development/selection.json) preserves the historical identity, timestamps and input/source hashes. Full development-search replay commands are in [modeling](modeling.md).

The independent GH200 capture then supplied **1,237 steady rows**. Each evaluated bundle remained fixed; no candidate was changed after this result:

| Frozen predictor | GH200 MAE (W) | GH200 RMSE (W) | GH200 R² |
| --- | ---: | ---: | ---: |
| Device-development winner: pooled workload mean | 31.0889 | 34.7136 | 0.0010 |
| Known-device forest, external-device challenge | 37.0065 | 42.9896 | -0.5322 |
| Historical fixed H100 linear model, reference replay | 23.0028 | 30.8382 | 0.2116 |

The historical linear model is better retrospectively on this capture, but it does not replace the frozen development winner. Two development GPUs did not establish unseen-device accuracy, and strong known-device trial validation did not transfer to GH200. The GH200 challenge tests a different objective from known-device prospective validation. Checked-in evidence: [frozen GH200 winner](../results/quality-2026-10-20/gh200/frozen-evaluation/evaluation.json), [forest challenge](../results/quality-2026-10-20/gh200/known-device-challenge/evaluation.json) and [historical reference](../results/quality-2026-10-20/gh200/transfer/evaluation.json). No post-test tuning or winner change occurred. Historical H100 trial-held-out and fixed H100→H200 results remain in their original reports.

## Prospective H200 capture: final known-device result

The selected forest froze at 20:40:29.710867 UTC, before the new H200 capture began at 20:41:24 UTC. The capture completed **24 verified workload windows**, **2,090 raw readings** and **1,153 steady rows**. Its UUID was already represented in development, while run/session IDs were disjoint and every test timestamp followed the selection freeze. The saved bundle was loaded without refitting or tuning. This is one prospective capture; its correlated rows are not 1,153 independent experiments.

| Predictor | Final H200 MAE (W) | RMSE (W) | R² |
| --- | ---: | ---: | ---: |
| Frozen known-device random forest | 2.3194 | 9.5560 | 0.9488 |
| Pooled development-workload mean | 18.7402 | 23.1573 | 0.6994 |
| Pooled development-global mean | 37.3955 | 42.3696 | -0.0063 |

The forest reduced RMSE **58.7%** against the workload baseline. All rows remain in the final score. Median absolute error was **0.815 W**, p95 **3.094 W**, and p99 **73.998 W**. There were **24 errors above 25 W**, mainly stencil samples; the ten largest residuals contributed **53.2%** of squared error. The tail remains visible below. No errors were trimmed and no subsequent model changes were made. The **3.108 W** selection score is development validation; **9.556 W** is the independent-session result.

![Independent H200 RMSE comparison and every untrimmed residual](../results/quality-2026-10-20/h200-independent/evaluation/validation.png)

[Final metrics](../results/quality-2026-10-20/h200-independent/evaluation/evaluation.json), [residual summary](../results/quality-2026-10-20/h200-independent/evaluation/residual-analysis.json) and [all predictions](../results/quality-2026-10-20/h200-independent/evaluation/predictions.csv) are curated evidence. Regenerate the separate [research figure gallery](../results/research-2026-10-20/README.md), preserving the original evidence archive, with:

```bash
python scripts/plot_research_results.py
```

## DGX Spark GB10: a failed external transfer

The additional native GB10 capture completed 24 verified workload windows, 1,905 raw readings and **1,023 steady rows**. Neither model selection nor fitting used this device. All final rows were retained; the same frozen bundles were evaluated without adaptation.

| Frozen predictor | GB10 MAE (W) | RMSE (W) | R² |
| --- | ---: | ---: | ---: |
| Cross-device development winner: workload mean | 388.2923 | 391.2667 | -183.6820 |
| Known-device random forest, external challenge | 378.7280 | 379.2773 | -172.5370 |
| Historical fixed H100 linear reference | 222.4800 | 250.1105 | -74.4646 |

All three failed on GB10. Its lower GPU-power envelope and missing memory-clock telemetry differ from H100/H200 development; missing values remain unavailable and are handled by the frozen training-fitted preprocessing, never replaced with invented measurements. This result does not isolate why transfer failed or justify selecting the historical reference after seeing the scores. No adaptive fitting, clipping, error removal or winner change followed the final evaluation.

The positive 9.556 W result remains limited to a prospective session on a known H200. The API's reference-identity guard is essential: successful capture and CUDA correctness on H100, H200, GH200 and GB10 do not approve a power model for all four devices. [Spark hardware/sensor scope](dgx-spark-validation.md) and the [curated archive](../results/quality-2026-10-20/README.md) retain separate collection and model evidence. A100 validation remains queued; no A100 accuracy claim is made.
