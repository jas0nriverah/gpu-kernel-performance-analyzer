# Data, serving and hardware validation — 20 October 2026

The [final publication check](../results/research-validation-2026-10-20.json) passed **254 CPU tests**, verified the curated archive and all research figure dependencies, and reproduced all **3,413** final forest predictions from checked-in training inputs. Earlier receipts below retain their original test counts and source snapshots.

The quality pass adds transactional SQL ingestion, a strict serving contract and native hardware preflight. Evidence below was measured locally. The complete working output is under ignored `outputs/quality-2026-10-20/`; selected receipts, metrics and captures are curated in [results/quality-2026-10-20](../results/quality-2026-10-20/README.md). Repository tests and scripts support reproduction. Published H100 measurements remain unchanged.

## SQL reliability

Five completed captures across H100, H200, GH200 and GB10, including the prospective H200 session, were ingested into `experiments.sqlite`:

| Check | Observed result |
| --- | --- |
| Runs / verified trial windows | 5 / 120 |
| Raw / steady telemetry rows | 10,301 / 5,689 |
| Artifact digest rows, including manifests | 269 |
| Repeated ingestion | `inserted: false` on repeated ingestion; no added rows |
| SQLite integrity / foreign keys | `ok` / no errors |

`warehouse-final-verification.json` records final counts and integrity checks; final SQL aggregates are retained alongside earlier count receipts in the curated warehouse directory. The SQL integrates consecutive steady samples with the trapezoidal rule and reproduces capture summaries. Fourteen focused tests cover archived summary parity, retries, conflicting identities, hash mismatch, duplicate samples, nonfinite values, insertion rollback, concurrent identical ingestion and nullable optional sensors without weakening required power validation. See [schema, validation and limitations](warehouse.md).

```bash
.venv/bin/python -m gpu_power_pipeline.warehouse ingest \
  --database /tmp/power-quality.sqlite \
  results/h100-power-2026-09-22 \
  outputs/prizepicks-h200-2026-10-20/power
.venv/bin/python -m gpu_power_pipeline.warehouse query \
  --database /tmp/power-quality.sqlite
```

## Serving behavior and HTTP smoke

The pinned H100 linear regression bundle is version `20261020T201154Z`. Replay in batches of 128 accepted all **1,123 H100 rows**, with **0 W** maximum absolute difference from direct saved-model predictions (`api-h100.json`). All **1,153 H200 rows** were rejected across ten batches with HTTP 422 for unseen GPU identity (`api-h200-rejection.json`). The earlier permissive H200 API result remains historical evidence in [the H200 review](h200-review.md). Offline transfer evaluation still reports its negative accuracy result; serving rejection does not change that score.

The API now enforces required feature values, numeric and physical bounds and a configurable batch limit; separate liveness/readiness endpoints and process-local failure counters make service failures visible. See [the serving contract](serving.md).

A real localhost HTTP smoke used one Uvicorn worker on the x86_64 H200 host, with CPU inference and the GPU unused. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were each 1. Prediction logging and access logging were disabled; the server listened on port 8765.

| Setting or result | Value |
| --- | --- |
| Measured requests / clients / records per batch | 100 / 4 / 16 |
| Warmups, excluded from measurement | 5 |
| Failed requests / warmups | 0 / 0 |
| p50 / p95 / p99 request latency | 24.47 / 29.78 / 31.55 ms |
| Successful request throughput | 162.2 requests/s |
| Measured elapsed time | 0.617 s |

`http-benchmark/summary.json`, `requests.csv` and `warmups.csv` retain measurements. This short smoke checks the real HTTP path; it is not sustained production capacity, a deployment SLO, or a cold-start study. Shared client/server resources and warm caches limit interpretation. The [serving guide](serving.md) gives the reproducible load-test command and response checks.

## Hardware and regression checks

The preflight checks native Python/compiler/runtime architecture, selected GPU identity, available memory and an idle allocation. The validation runner records preflight/build/source metadata, builds into a fresh directory and offers `MODE=smoke` (18 edge configurations) or `MODE=full` (baseline, three timing sweeps and edges). A new H200 `sm_90` build passed all 18 edge cases; evidence is in `h200-smoke/`. This validates the H200 path; portable checks alone cannot establish behavior on a different host architecture.

```bash
PYTHON=.venv/bin/python MODE=smoke \
  OUT_DIR=outputs/quality-replay/h200-smoke \
  bash scripts/cross_validate_gpu.sh
```

Native Arm GH200 validation subsequently completed 24 baseline configurations, three 74-configuration sweeps (**11,100 scaling timing samples**) and 18 edge cases, all verified. The [GH200 guide](gh200-validation.md) records its environment and power capture. Timing artifacts are under `gh200/timing/`.

Existing comparison commands matched all 74 scenarios for each corresponding trial across H100, H200 and GH200. At a 5% slowdown threshold, H100→GH200 had 0/0/0 slower scenarios across trials; H200→GH200 had 8/6/9. Every cross-device gate was ineligible. `gh200/comparison/` retains nine reports and a summary. Device, host architecture, compiler, date and environment differ; these descriptive ratios cannot isolate a hardware effect or establish significance.

An earlier full CPU suite passed **204 tests in 93.52 seconds**; after model-development and evaluation coverage, a later full run passed **253 tests**. After the final huge-integer input rejection fix, **25 focused API tests** passed; these focused receipts remain separate from full-suite counts. Ruff passed. The runtime-preflight correction passed **12 focused checks** and the fresh H200 smoke. Tests exercise malformed data and service failures; CPU CI does not measure CUDA performance. Longer service load, deployment rollback and additional independent collection sessions remain future validation.

## Frozen model development and independent evaluation

H100/H200 became development data for a fixed twelve-candidate search. Mean bidirectional device-held-out RMSE selected a workload-mean baseline (**25.4766 W**); the best learned candidate scored **33.7206 W**. Separately, known-device whole-trial validation selected a 64-tree random forest (**3.1080 W** mean, **5.5052 W** worst-fold RMSE). Those are selection/validation scores, not independent test accuracy. The known-device forest froze at 20:40:29.710867 UTC; a fresh H200 capture started afterward. Its 24 verified workload windows yielded 1,153 steady rows, all retained: **9.5560 W RMSE** versus **23.1573 W** for the workload baseline (**58.7% reduction**), with **2.3194 W MAE** and **0.9488 R²**. This is one prospective known-device session, not an unseen-GPU test. Twenty-four rows exceeded 25 W error; the p99 absolute error was 73.998 W. [Curated final evidence](../results/quality-2026-10-20/h200-independent/evaluation/evaluation.json) and [untrimmed residual figure](../results/quality-2026-10-20/h200-independent/evaluation/validation.png) expose the tail.

On 1,237 independent GH200 steady rows, the frozen cross-device baseline scored **34.7136 W RMSE**, the known-device forest scored **42.9896 W**, and the historical fixed H100 linear reference scored **30.8382 W**. The retrospective reference ranking does not change the frozen winner. These results expose the limits of two-device development; no unseen-GPU accuracy claim follows from the low known-device validation error. [Protocol, full metrics and evidence paths](model-generalization.md).

Additional A100 validation is queued (PACE job 6426654). DGX Spark native GB10 completed 24 baseline configurations, three 74-configuration sweeps and 18 edge cases; its power capture completed 24 verified windows, 1,905 raw rows and 1,023 steady rows (job 11546, compute capability 12.1). Across four GPUs, the scaling sweeps retain 44,400 timing samples. See [Spark methodology](dgx-spark-validation.md). New H200 API replay accepted all 1,153 prospective-capture rows with **0 W** difference from direct saved-model predictions. Timing correctness, model accuracy and serving parity are separate checks.

Both new H100/H200-trained bundles rejected all **1,023 Spark rows** in eight batches each with HTTP 422 for an unapproved GPU, serving zero predictions. The corresponding GH200 checks rejected all **1,237 rows** per bundle. These checks exercise the device guard on real external telemetry, including Spark's missing memory-clock readings; offline scores remain unchanged. Curated receipts are in `spark/api-rejection.json`, `spark/api-forest-rejection.json` and the corresponding `gh200/` files under the quality results archive.

DGX Spark GB10 final transfer failed: frozen workload-mean **391.2667 W RMSE**, known-device forest **379.2773 W**, and historical H100 linear reference **250.1105 W**. Its lower power envelope and unavailable memory-clock sensor differ from development; those observations do not isolate a causal explanation. No adaptation or test-driven winner change occurred. Model serving remains restricted to reference GPU identities. [Full transfer scope](model-generalization.md) and [Spark collection evidence](dgx-spark-validation.md) distinguish successful hardware collection from model portability. The temporary SSH key was revoked and the Spark allocation released; pre-existing user jobs were unchanged. A100 job 6426654 and its dependent CPU job 6426680 remain queued.
