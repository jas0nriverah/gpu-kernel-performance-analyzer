from pathlib import Path

import pytest

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.persistence import ModelMetadata, new_version_id, save_model_bundle
from gpu_power_pipeline.train import fit_full_pipeline

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from gpu_power_pipeline.api import create_app  # noqa: E402


def _seed_registry(tmp_path: Path, model_name: str = "random_forest") -> Path:
    df = load_dataset(source="synthetic", n_samples=400, random_state=1)
    pipeline, numeric, categorical = fit_full_pipeline(df, model_name, random_state=1)
    metadata = ModelMetadata(
        model_name=model_name,
        target="power_watts",
        numeric_features=numeric,
        categorical_features=categorical,
        feature_order=numeric + categorical,
        metrics={"rmse": 1.0},
        split_strategy="time",
        source="synthetic",
        version=new_version_id(),
    )
    save_model_bundle(pipeline, metadata, registry_dir=tmp_path / "artifacts")
    return tmp_path / "artifacts"


def test_health_and_predict(tmp_path: Path):
    registry = _seed_registry(tmp_path)
    app = create_app(registry_dir=str(registry), model_name="random_forest")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    payload = {
        "records": [
            {
                "gpu_utilization_pct": 70,
                "memory_utilization_pct": 60,
                "graphics_clock_mhz": 1500,
                "memory_clock_mhz": 5000,
                "temperature_c": 65,
                "ambient_c": 23,
                "workload_type": "training",
            }
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_power_watts"] > 0


def test_health_degraded_when_no_model(tmp_path: Path):
    app = create_app(registry_dir=str(tmp_path / "empty"), model_name="random_forest")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert client.get("/ready").status_code == 503
    assert client.get("/live").status_code == 200
    # Predict should fail clearly when no model is available.
    resp = client.post("/predict", json={"records": [{"gpu_utilization_pct": 50}]})
    assert resp.status_code == 503


@pytest.fixture
def serving(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from gpu_power_pipeline import api

    model = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="test",
            version="pinned",
            feature_order=["gpu_name", "gpu_utilization_pct", "graphics_clock_mhz"],
            numeric_features=["gpu_utilization_pct", "graphics_clock_mhz"],
            categorical_features=["gpu_name"],
            reference_profile={"categorical": {"gpu_name": {"categories": ["H200"]}}},
        ),
        predict=lambda records: np.full(len(records), 200.0),
    )
    loads = []

    def load(*args, **kwargs):
        loads.append(1)
        return model

    monkeypatch.setattr(api, "load_model", load)
    return api, model, loads


def _record():
    return {"gpu_name": "H200", "gpu_utilization_pct": 70, "graphics_clock_mhz": 1500}


@pytest.mark.parametrize(
    "patch",
    [
        {"gpu_name": "A100"},
        {"gpu_name": " "},
        {"gpu_name": None},
        {"gpu_utilization_pct": None},
        {"gpu_utilization_pct": "nan"},
        {"gpu_utilization_pct": float("inf")},
        {"gpu_utilization_pct": -1},
        {"gpu_utilization_pct": 101},
        {"gpu_utilization_pct": True},
        {"graphics_clock_mhz": 0},
    ],
)
def test_contract_rejects_invalid_values(serving, patch):
    import json

    api, _, _ = serving
    client = TestClient(api.create_app())
    # Raw JSON permits testing NaN/Infinity parser behavior as well.
    response = client.post(
        "/predict",
        content=json.dumps({"records": [{**_record(), **patch}]}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert client.get("/metrics").json()["prediction_failures"] == 1


def test_batch_and_missing_fields(serving):
    api, _, loads = serving
    client = TestClient(api.create_app(max_batch_size=1))
    for records in [[], [{}], [_record(), _record()]]:
        assert client.post("/predict", json={"records": records}).status_code == 422
    for _ in range(2):
        response = client.post("/predict", json={"records": [_record()]})
        assert response.status_code == 200
        assert response.json()["model_version"] == "pinned"
    assert len(loads) == 1
    assert client.get("/live").status_code == 200
    assert client.get("/ready").status_code == 200


@pytest.mark.parametrize("result", [float("nan"), float("inf"), "error"])
def test_inference_failures_are_server_errors(serving, result):
    api, model, _ = serving

    def predict(records):
        if result == "error":
            raise ValueError("internal failure")
        return [result]

    model.predict = predict
    client = TestClient(api.create_app())
    assert client.post("/predict", json={"records": [_record()]}).status_code == 500
    assert client.get("/metrics").json()["prediction_failures"] == 1


def test_log_failures_observable(serving, monkeypatch, tmp_path):
    api, model, _ = serving
    monkeypatch.setenv("GPU_POWER_PREDICTION_LOG", str(tmp_path / "log"))
    model._align_frame = lambda records: records

    def fail(**kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(api, "append_prediction_log", fail)
    client = TestClient(api.create_app())
    assert client.post("/predict", json={"records": [_record()]}).status_code == 200
    metrics = client.get("/metrics").json()
    assert metrics["log_write_failures"] == 1
    assert metrics["last_log_error"] == "disk unavailable"


@pytest.mark.parametrize("device", [None, "", " ", "A100", 123])
def test_portable_model_still_requires_approved_device(serving, device):
    api, model, _ = serving
    model.metadata.feature_order.remove("gpu_name")
    model.metadata.categorical_features = []
    client = TestClient(api.create_app())
    record = _record()
    if device is None:
        record.pop("gpu_name")
    else:
        record["gpu_name"] = device
    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422
    assert "gpu_name" in response.json()["detail"]
    accepted = client.post("/predict", json={"records": [_record()]})
    assert accepted.status_code == 200
    assert accepted.json()["predictions"][0]["predicted_power_watts"] == 200.0


def test_legacy_portable_model_without_device_profile_keeps_contract(serving):
    api, model, _ = serving
    model.metadata.feature_order.remove("gpu_name")
    model.metadata.categorical_features = []
    model.metadata.reference_profile = {}
    record = _record()
    record.pop("gpu_name")
    client = TestClient(api.create_app())
    assert client.post("/predict", json={"records": [record]}).status_code == 200


def test_portable_model_advertises_device_serving_requirement(serving):
    api, model, _ = serving
    meta = model.metadata
    meta.feature_order.remove("gpu_name")
    meta.categorical_features = []
    meta.target = "power_watts"
    meta.metrics = {}
    meta.split_strategy = "device_holdout"
    meta.source = "synthetic"
    meta.created_utc = "2026-10-20T00:00:00Z"
    client = TestClient(api.create_app())
    response = client.get("/model")
    assert response.status_code == 200
    metadata = response.json()
    assert "gpu_name" not in metadata["feature_order"]
    assert metadata["required_fields"] == [*meta.feature_order, "gpu_name"]
    assert metadata["allowed_gpu_names"] == ["H200"]
    record = {field: _record()[field] for field in metadata["required_fields"]}
    assert client.post("/predict", json={"records": [record]}).status_code == 200


def test_huge_json_integer_is_client_error_and_service_recovers(serving):
    api, _, _ = serving
    client = TestClient(api.create_app())
    record = {**_record(), "gpu_utilization_pct": 10**400}
    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422
    assert "finite number" in response.json()["detail"]
    assert client.get("/metrics").json()["prediction_failures"] == 1
    assert client.post("/predict", json={"records": [_record()]}).status_code == 200
