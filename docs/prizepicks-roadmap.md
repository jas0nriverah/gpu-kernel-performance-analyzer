# PrizePicks portfolio direction

Reviewed October 20, 2026, against this repository's integrated performance and power code.

The current [research report](research-report.md) and [model card](model-card.md) lead with the completed ML workflow: verified CUDA telemetry, transactional SQL, frozen model selection, prospective validation and a versioned API. A frozen forest achieved **9.56 W RMSE**, **58.7% below** the workload baseline, on one later H200 session. Complete residuals and failed GH200/GB10 transfers remain visible. This supports known-device session validation, not general GPU portability. The [historical H200 review](h200-review.md) preserves the earlier fixed H100 model's negative transfer result.

## What the roles ask for

The [Machine Learning Internship — Spring 2027](https://www.prizepicks.com/position?gh_jid=8038883003)
emphasizes Python, SQL, reliable sports data streams, and moving models into stable, scalable services.
GCP, deployment automation, databases, MLOps, and lower-level languages are differentiators.
The posting was returned by PrizePicks' public Greenhouse API on the review date.

The [Data & ML Platform internship description](https://www.choppingblock.ai/jobs/data-machine-learning-platform-engineering-internship-spring-2027-at-prizepicks)
also emphasizes distributed ETL, Kubernetes, and CPU/GPU architecture. Its original job ID,
8009261003, returned HTTP 404 and was absent from the live employer job feed on the review date.
Use its requirements for portfolio alignment; do not assume the role is still accepting applications.
Both descriptions have a Spring 2027 title but a 2026 program-date line; confirm dates with recruiting.

## Existing evidence and actual gaps

| Hiring signal | Already present | Most valuable next evidence |
| --- | --- | --- |
| Python and C++ | Correctness-verified H100/H200 and native Arm GH200/GB10 captures; 44,400 scaling timing samples | Additional independent sessions and controlled same-device regression studies |
| Data reliability | Provenance plus transactional SQLite ingestion, constraints, idempotence and rollback tests | Additional independent captures and deployment-specific data handling |
| Applied ML | Train-only preprocessing, twelve fixed candidates, prospective H200 validation and negative GH200/GB10 tests | Held-out workload families, more devices and additional untouched temporal captures |
| Model operations | Pinned loaded versions, strict FastAPI contracts, readiness and error counters, short HTTP smoke | Sustained load, deployment rollback and operational monitoring |
| Deployment | CPU test/lint GitHub Actions | Built and tested container; a recorded deployment and rollback |
| Kubernetes/GCP | No deployment evidence found | Run a small local Kubernetes demo, then deploy a CPU inference service to GCP if desired |
| Sports context | No sports dataset or task | A small separately scoped sports-model example using the same serving and evaluation practices |

The strongest framing for the existing project is **GPU telemetry ML and performance infrastructure**.
The H200 generates real observations and supports systems experiments. The tabular models can train
and serve on CPU; using a neural network solely to consume the H200 would not establish better ML work.

## Recommended implementation order

### 1. Extend completed model validation

The historical transfer experiment fixed its model from the published H100 study before examining H200 targets. It fitted preprocessing and weights only on H100, reported training-only baselines and retained the negative result. Later development explicitly reclassified H200 as development data and reserved other captures for final evaluation.

The review adds `python -m gpu_power_pipeline.transfer` for this experiment. It rejects overlapping
GPU UUIDs, records input/model hashes, reloads the saved bundle, and emits drift and error reports.
The default linear model comes from the prior H100 result; H200 is not used for model selection.
Cross-device results are also confounded by capture dates and environments.

A later bounded H100/H200 development search selected a workload-mean baseline at 25.4766 W mean device-held-out RMSE. The best learned candidate scored 33.7206 W. On untouched GH200 rows the frozen baseline scored 34.7136 W RMSE, while a separately selected known-device forest scored 42.9896 W. A historical H100 linear reference scored 30.8382 W retrospectively; it does not replace the frozen winner. Two development GPUs did not establish general portability.

The known-device forest achieved 3.1080 W mean validation RMSE across six whole-trial folds, then froze before a new H200 collection began. On the prospectively collected H200 session it achieved **9.5560 W RMSE**, a **58.7%** reduction versus the 23.1573 W workload baseline. This retained all 1,153 correlated telemetry rows across 24 windows; 24 errors exceeded 25 W and p99 absolute error was 73.998 W. It establishes evidence for one future session on a known GPU, while the GH200 challenge still exposes poor portability. See [the two objectives and frozen protocol](model-generalization.md).

Next, collect additional independent sessions and workload shapes. Split by whole sessions,
withhold workload families/sizes, select models on validation folds, and evaluate once on a final
test set. Keep future H200 sessions untouched if adapting to the first H200 capture. Add
train-only feature selection to the historical BMC path before presenting that dataset as formal evidence.
The legacy `train.py` full-data serving/refit path does not turn its earlier holdout score into an independent test of the refitted artifact. The newer frozen known-device forest has a separate prospective H200 evaluation after its selection/refit; its 9.556 W result belongs to that final bundle.

### 2. SQL ingestion implemented; extend only as needed

The [SQLite warehouse](warehouse.md) stores runs, artifact hashes, trial windows and raw samples,
with foreign keys, identity/value checks and schema versioning. Completed captures are ingested
atomically; identical retries add no rows and conflicting identities fail. Invalid input is rejected
with an error, leaving source files available for investigation. SQL reproduces steady-power means
using trapezoidal integration. The recorded H100/H200/GH200/GB10 check, including a prospective H200 session, contains 5 runs, 120 windows, 10,301 raw
samples and 5,689 steady samples; integrity and foreign-key checks pass.

The current schema deliberately omits prediction joins and a database server. Add those when an
actual evaluation or concurrent-service workload needs them. No ingestion-rate claim was measured.

### 3. Serving contracts implemented; sustain and deploy next

The API now requires the loaded feature contract, rejects nonfinite/out-of-range values and unseen
GPU names, bounds batches, separates liveness/readiness and exposes failure/logging counters.
The historical H100-only model accepts all 1,123 H100 replay rows and rejects H200. The later pooled H100/H200 forest serves all 1,153 prospective H200 rows with exact saved-model parity. Both pooled selected bundles reject all 1,023 GB10 rows for unknown GPU identity. Offline evaluation preserves those devices’ negative transfer results separately.

A real localhost HTTP smoke exercised 100 requests, four clients and batches of 16 with zero errors.
It lasted only 0.617 seconds, so it is functional measurement rather than sustained capacity or an
achieved production latency target. [Quality validation](quality-validation.md) records settings,
latencies and local evidence paths. Next measure longer runs, cold starts, resource saturation and
version rollback in the chosen deployment environment.

### 4. Demonstrate one deployment path

Build a container and reproduce training-to-registry-to-serving from a clean environment. CI should
check schema and prediction contracts, build the image, and smoke-test a pinned model artifact.
Use a small local Kubernetes deployment with readiness probes and CPU/memory requests to demonstrate
platform concepts. Separate GPU collection jobs from the CPU model service.

A later GCP deployment can demonstrate cloud delivery, provided it is actually executed and measured.
Keep cloud setup out of résumé claims until there are deployment logs, a health check, and a recorded rollback.
Do not add Kafka, Spark, or a multi-node cluster without a workload that needs them.

### 5. Add sports context as a bounded example

For the ML internship, consider a small NBA player-stat prediction example backed by a documented,
licensed historical dataset. Use SQL to build rolling features from earlier games only; retain
event and availability timestamps; split chronologically and compare against a rolling-average baseline.
If predicting probabilities, evaluate calibration as well as accuracy.

Serve that model through the same deployment practices and replay historical events to test freshness,
duplicates, and late arrivals. Label replay as replay. This establishes relevant domain interest
without claiming access to PrizePicks data or inferring profitable betting performance.

## Résumé evidence rules

Lead with the problem, implementation, and measured outcome. Use at most three project bullets.
For the ML application, prioritize evaluation, feature/data handling, and serving. For the platform
application, prioritize reliability, SQL, deployment, and CPU/GPU performance.

Existing published evidence supports statements such as:

- Built a Python/CUDA experiment pipeline covering six kernels and 74 configurations; retained
  11,100 H100 timing samples with correctness checks and source/binary provenance.
- Evaluated GPU board-power models using entire held-out H100 trials; linear regression achieved
  3.11 W mean RMSE across three folds on 1,123 measured telemetry samples.
- Implemented versioned model persistence, batch and FastAPI inference, and feature-drift reports.

The H100 score measures repeat-session behavior on one GPU. It is not a cross-device score.
The API exists locally; it is not evidence of a cloud deployment or an achieved latency target.
SQL reliability and strict API contract checks now have recorded evidence. Cloud deployment, sustained service capacity and latency targets remain unproven. The short HTTP smoke should not become an SLO résumé claim.

Related evidence: [H100 model report](../results/h100-power-model-2026-09-22/REPORT.md),
[engineering decisions](engineering.md), [power methodology](power_measurement.md).

## Current application bullets

For an ML application, use measured evaluation scope explicitly:

- Built a GPU board-power evaluation pipeline with train-only preprocessing, twelve fixed candidates and separate known-device and device-held-out objectives; retained the negative independent GH200 result and froze final artifacts before evaluation.
- Selected and froze a 64-tree GPU board-power random forest before a new H200 session; achieved 9.56 W RMSE across 24 verified workload windows, reducing error 58.7% versus a training-workload-mean baseline while retaining all telemetry and its residual tail.
- Implemented bounded FastAPI prediction contracts, readiness and failure metrics, with exact saved-model parity on 1,153 prospective H200 telemetry rows and rejection of unseen GPU identities.

For a platform application:

- Built transactional SQLite ingestion for five measured captures across four GPUs: 10,301 raw telemetry rows and 120 verified workload windows, with artifact hashes, identity constraints, idempotent retries and tested rollback.
- Validated Python/CUDA execution on H100, H200 and native Arm GH200/GB10, retaining 11,100 scaling timing samples per device and descriptive comparisons that reject cross-device CI gates.
- Added native hardware preflight, fresh-build capture workflows and version-pinned CPU inference with explicit readiness and malformed-input checks.

These bullets describe local evidence. They do not claim cloud deployment, production SLOs, unseen-GPU accuracy or sports-data experience. A100 validation remains queued. DGX Spark GB10 timing and power collection passed, but final model transfer failed (391.27 W RMSE for the frozen workload baseline; 379.28 W for the known-device forest). Four-device pipeline compatibility does not imply model portability or permission to serve unknown GPUs.
