# Serving contract and local HTTP measurement

Reconstruct the archived known-device forest on CPU, then start the API with the resulting registry. Model weights are deliberately omitted from the repository; the replay verifies the recorded inputs/configuration and archived predictions. Use a new output directory:

```bash
python -m pip install -e '.[power,api]'
python scripts/reproduce_frozen_model.py --outdir outputs/model-replay
GPU_POWER_REGISTRY_DIR=outputs/model-replay/registry \
  GPU_POWER_MODEL_NAME=rf_unbounded_full GPU_POWER_MODEL_VERSION=latest \
  python -m uvicorn gpu_power_pipeline.api:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/docs` or inspect `/model` before constructing requests. This is retrospective model reproduction and local serving, not a new prospective experiment. [Model card](model-card.md) · [Research report](research-report.md).

Each application instance loads its bundle once. `latest` resolves at creation and the resulting version stays pinned until restart. If loading fails, the instance remains unavailable until restarted. A newly published bundle does not change an existing instance.

| Endpoint | Behavior |
| --- | --- |
| `GET /live` | HTTP 200 while the application can answer requests. |
| `GET /ready` | HTTP 200 with model identity when loaded; HTTP 503 with degraded health otherwise. |
| `GET /health` | Same response body and readiness behavior as `/ready`; degraded instances return 503. |
| `GET /model` | Loaded metadata including exact `feature_order`, API `required_fields`, and version; HTTP 503 if unavailable. |
| `POST /predict` | Accepts `{"records": [...]}` and returns model identity plus indexed power predictions in watts. |
| `GET /metrics` | Process-local request, failure, successful-record, and log-failure counters plus the last log error. |

Every prediction record must contain every entry in `/model`'s `required_fields`. This includes predictive `feature_order` plus `gpu_name` when the saved profile enables the GPU guard, even if GPU name is not a predictive feature. Required values cannot be null. Numeric values must be finite JSON numbers; numeric strings and booleans are rejected. Percentage features ending in `_pct` must be between 0 and 100 inclusive. Clock features and `memory_total_*` capacities must be positive. Celsius features ending in `_c` must be at least -273.15. Categorical features must be nonblank strings. Extra fields are ignored by prediction.

When metadata contains `reference_profile.categorical.gpu_name.categories`, serving accepts only those listed GPU names. This protects against silently deploying a model on an unseen GPU. A missing profile provides no GPU protection, and a truncated category list may reject GPUs omitted from that list. The tolerant offline `PowerModel.predict` interface remains available for intentional transfer evaluation; API rejection is not evidence of transfer accuracy.

Batches must contain at least one record and at most 512 by default. Set `GPU_POWER_MAX_BATCH_SIZE` to a positive integer, or pass `max_batch_size` to `create_app`, to change the limit. This bounds records processed in one inference request; HTTP body size and concurrency still require deployment controls.

Invalid input returns HTTP 422. An unavailable model returns HTTP 503. Unexpected inference failures or nonfinite/malformed model output return HTTP 500. Prediction responses retain `model_name`, `model_version`, and `predictions` with `index` and `predicted_power_watts`.

Set `GPU_POWER_PREDICTION_LOG` for optional JSONL prediction logging. A write failure does not fail a successful prediction: `/metrics` increments `log_write_failures` and records `last_log_error`. The counters `prediction_requests`, `prediction_failures`, and `prediction_records` count prediction attempts, failed attempts, and successfully predicted rows. They reset on restart and are separate for every worker; these are neither durable nor distributed metrics. The last log error remains available after subsequent successful requests.

## Local HTTP microbenchmark

With the service already running, measure its actual localhost HTTP path using local `nvidia-smi` telemetry:

```bash
.venv/bin/python scripts/load_test_api.py \
  --url http://127.0.0.1:8000 \
  --telemetry results/quality-2026-10-20/h200-independent/power/steady_telemetry.csv \
  --requests 100 --concurrency 4 --batch-size 16 --warmups 5 \
  --outdir outputs/http-load-test
```

The output directory must be new or empty. The script normalizes telemetry through `load_dataset("local_nvidia_smi")`, selects `/model.required_fields`, including any GPU guard field, and cycles rows to fill batches. It checks model name, version, row count, indices, and finite predictions for every successful HTTP response. HTTP errors, timeouts, and response mismatches are recorded as failures; any measured or warmup failure produces a nonzero exit code.

`requests.csv` and `warmups.csv` preserve per-request latency, HTTP status, attempted rows, success, and error text. `summary.json` reports all-request and successful-request p50/p95/p99 latency separately, measured wall time, successful request/record throughput, errors, client environment, and model identity. Warmups are excluded from measured latency and throughput. Client concurrency uses a fixed thread pool and HTTP connection limits.

This is a local HTTP microbenchmark, not production capacity evidence. Shared client/server resources, telemetry repetition, logging, warm caches, worker count, and host contention affect the measurements. Capture server settings and hardware alongside these files when comparing runs. It does not establish power-model accuracy or GPU portability.
