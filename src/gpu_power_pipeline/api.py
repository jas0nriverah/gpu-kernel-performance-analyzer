"""FastAPI inference service for power predictions.

Loads a persisted model bundle from the artifact registry and exposes:

- ``GET /health``  - readiness + loaded model identity
- ``GET /model``   - metadata describing the loaded model
- ``POST /predict`` - batch power predictions from telemetry records

Configure via environment variables (with sensible defaults):

- ``GPU_POWER_REGISTRY_DIR`` (default ``artifacts``)
- ``GPU_POWER_MODEL_NAME``   (default ``random_forest``)
- ``GPU_POWER_MODEL_VERSION`` (default ``latest``)
- ``GPU_POWER_PREDICTION_LOG`` (optional JSONL prediction log path)
- ``GPU_POWER_MAX_BATCH_SIZE`` (default ``512``)
"""

from __future__ import annotations

import math
import os
from numbers import Real
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field

from .inference import PowerModel, load_model
from .monitoring import append_prediction_log


class TelemetryRecord(BaseModel):
    """A single telemetry observation. Unknown fields are allowed and ignored
    unless the loaded model expects them."""

    model_config = ConfigDict(extra="allow")


class PredictRequest(BaseModel):
    records: List[TelemetryRecord] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    index: int
    predicted_power_watts: float


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None


def validate_records(model: PowerModel, records: List[Dict[str, Any]]) -> None:
    """Enforce the serving contract; offline prediction remains tolerant.

    Percentages must be in [0, 100], clock/memory capacities positive,
    and Celsius temperatures no lower than absolute zero.
    """
    meta = model.metadata
    categorical_profile = meta.reference_profile.get("categorical", {})
    gpu_profile = categorical_profile.get("gpu_name", {})
    known_gpus = gpu_profile.get("categories", [])
    for index, record in enumerate(records):
        # Device approval is a serving constraint even when a portable model
        # deliberately excludes device identity from its predictive features.
        if "gpu_name" in categorical_profile:
            device = record.get("gpu_name")
            if not isinstance(device, str) or not device.strip():
                raise HTTPException(status_code=422, detail=f"records[{index}].gpu_name must be a nonblank string")
            if device not in known_gpus:
                raise HTTPException(
                    status_code=422,
                    detail=f"records[{index}].gpu_name is not a GPU represented in the model reference profile",
                )
        for feature in meta.feature_order:
            value = record.get(feature)
            error = None
            if value is None:
                error = "is required and cannot be null"
            elif feature in meta.numeric_features:
                try:
                    finite = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    error = "must be a finite number"
                elif feature.endswith("_pct") and not 0 <= value <= 100:
                    error = "must be between 0 and 100"
                elif ("clock" in feature or feature.startswith("memory_total_")) and value <= 0:
                    error = "must be positive"
                elif feature.endswith("_c") and value < -273.15:
                    error = "must be at least -273.15 Celsius"
            elif feature in meta.categorical_features:
                if not isinstance(value, str) or not value.strip():
                    error = "must be a nonblank string"
            if error:
                raise HTTPException(status_code=422, detail=f"records[{index}].{feature} {error}")


def create_app(
    registry_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    max_batch_size: Optional[int] = None,
) -> FastAPI:
    """Build the FastAPI app, attempting to load the configured model."""
    registry_dir = registry_dir or os.environ.get("GPU_POWER_REGISTRY_DIR", "artifacts")
    model_name = model_name or os.environ.get("GPU_POWER_MODEL_NAME", "random_forest")
    version = version or os.environ.get("GPU_POWER_MODEL_VERSION", "latest")
    prediction_log = os.environ.get("GPU_POWER_PREDICTION_LOG")
    if max_batch_size is None:
        max_batch_size = int(os.environ.get("GPU_POWER_MAX_BATCH_SIZE", "512"))
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")

    app = FastAPI(
        title="GPU Power Modeling API",
        version="0.1.0",
        description="Predict power (watts) from telemetry using a trained model bundle.",
    )

    state: Dict[str, Any] = {"model": None, "error": None}
    counters = {"prediction_requests": 0, "prediction_failures": 0, "prediction_records": 0, "log_write_failures": 0}
    counter_lock = Lock()

    def increment(key: str, count: int = 1) -> None:
        with counter_lock:
            counters[key] += count

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        if request.url.path == "/predict":
            increment("prediction_requests")
            increment("prediction_failures")
        return await request_validation_exception_handler(request, exc)

    def _try_load() -> Optional[PowerModel]:
        try:
            model = load_model(registry_dir, model_name, version=version)
            state["model"] = model
            state["error"] = None
            return model
        except Exception as exc:  # noqa: BLE001 - surfaced via /health and 503s
            state["model"] = None
            state["error"] = str(exc)
            return None

    _try_load()

    def _get_model() -> PowerModel:
        model = state["model"]
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded: {state.get('error') or 'unavailable'}",
            )
        return model

    @app.get("/live")
    def live() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        # These counters belong to this process and reset when it restarts.
        with counter_lock:
            snapshot = dict(counters)
        return {"scope": "process", **snapshot, "last_log_error": state.get("prediction_log_error")}

    @app.get("/ready", response_model=HealthResponse)
    @app.get("/health", response_model=HealthResponse)
    def health(response: Response) -> HealthResponse:
        model = state["model"]
        if model is None:
            response.status_code = 503
        return HealthResponse(
            status="ok" if model is not None else "degraded",
            model_loaded=model is not None,
            model_name=model.metadata.model_name if model else None,
            model_version=model.metadata.version if model else None,
        )

    @app.get("/model")
    def model_info() -> Dict[str, Any]:
        model = _get_model()
        meta = model.metadata
        gpu_profile = meta.reference_profile.get("categorical", {}).get("gpu_name")
        required_fields = list(meta.feature_order)
        if gpu_profile is not None and "gpu_name" not in required_fields:
            required_fields.append("gpu_name")
        return {
            "model_name": meta.model_name,
            "version": meta.version,
            "target": meta.target,
            "feature_order": meta.feature_order,
            "required_fields": required_fields,
            "allowed_gpu_names": gpu_profile.get("categories", []) if gpu_profile is not None else None,
            "metrics": meta.metrics,
            "split_strategy": meta.split_strategy,
            "source": meta.source,
            "created_utc": meta.created_utc,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        increment("prediction_requests")
        try:
            model = _get_model()
            if len(request.records) > max_batch_size:
                raise HTTPException(status_code=422, detail=f"Batch exceeds {max_batch_size} records")
            records = [r.model_dump(exclude_unset=True) for r in request.records]
            validate_records(model, records)
            preds = np.asarray(model.predict(records), dtype=float)
            if preds.shape != (len(records),) or not np.isfinite(preds).all():
                raise RuntimeError("Model returned invalid predictions")
        except HTTPException:
            increment("prediction_failures")
            raise
        except Exception as exc:  # noqa: BLE001 - unexpected inference failure
            increment("prediction_failures")
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        increment("prediction_records", len(records))
        if prediction_log:
            try:
                append_prediction_log(
                    log_path=prediction_log,
                    model_name=model.metadata.model_name,
                    model_version=model.metadata.version,
                    input_frame=model._align_frame(records),
                    predictions=preds,
                )
            except Exception as exc:  # noqa: BLE001 - logging should not block inference
                increment("log_write_failures")
                state["prediction_log_error"] = str(exc)
        return PredictResponse(
            model_name=model.metadata.model_name,
            model_version=model.metadata.version,
            predictions=[PredictionItem(index=i, predicted_power_watts=float(p)) for i, p in enumerate(preds)],
        )

    return app


app = create_app()
