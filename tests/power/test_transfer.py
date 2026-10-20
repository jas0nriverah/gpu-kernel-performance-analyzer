"""Guard the independence of the external-device evaluation."""

import numpy as np
import pandas as pd
import pytest

from gpu_power_pipeline.transfer import baseline_predictions, validate_device_holdout


def test_rejects_overlapping_and_missing_device_identity():
    train = pd.DataFrame({"gpu_uuid": ["GPU-train"]})
    with pytest.raises(ValueError, match="overlap"):
        validate_device_holdout(train, train)
    with pytest.raises(ValueError, match="overlap"):
        validate_device_holdout(train, pd.DataFrame({"gpu_uuid": [" GPU-train "]}))
    for identity in (None, "", "   "):
        with pytest.raises(ValueError, match="missing"):
            validate_device_holdout(train, pd.DataFrame({"gpu_uuid": [identity]}))
    with pytest.raises(ValueError, match="gpu_uuid"):
        validate_device_holdout(train, pd.DataFrame({"power_watts": [100]}))
    validate_device_holdout(train, pd.DataFrame({"gpu_uuid": ["GPU-test"]}))


def test_baselines_never_read_test_targets_and_handle_unseen_workload():
    train = pd.DataFrame({"workload_type": ["a", "a", "b"], "power_watts": [100., 200., 300.]})
    test = pd.DataFrame({"workload_type": ["a", "b", "unseen"], "power_watts": [-999., 9999., 0.]})
    actual = baseline_predictions(train, test)
    np.testing.assert_allclose(actual["training_mean"], [200., 200., 200.])
    np.testing.assert_allclose(actual["training_workload_mean"], [150., 300., 200.])
    test["power_watts"] = np.nan
    changed = baseline_predictions(train, test)
    np.testing.assert_array_equal(actual["training_workload_mean"], changed["training_workload_mean"])


def _telemetry_fixture(uuid, rows=24):
    x = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "gpu_uuid": uuid,
        "gpu_utilization_pct": x * 3,
        "memory_utilization_pct": x * 2,
        "temperature_c": 30 + x,
        "workload_type": np.where(x % 2, "compute", "memory"),
        "power_watts": 50 + 4 * x,
    })


def test_transfer_bundle_predictions_and_training_are_independent_of_test_targets(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from gpu_power_pipeline.inference import PowerModel
    from gpu_power_pipeline.transfer import evaluate_transfer

    train = _telemetry_fixture("GPU-train")
    test = _telemetry_fixture("GPU-test", rows=6)
    test["workload_type"] = ["compute", "unseen", "memory"] * 2
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    first_dir = tmp_path / "first"
    first = evaluate_transfer(train_path, test_path, first_dir)
    rows = pd.read_csv(first_dir / "predictions.csv")
    model = PowerModel.from_dir(first["saved_model"])
    np.testing.assert_allclose(rows["linear_regression_predicted_w"], model.predict(test))
    np.testing.assert_allclose(rows["training_mean_predicted_w"], train.power_watts.mean())
    assert rows.loc[1, "training_workload_mean_predicted_w"] == train.power_watts.mean()
    assert first["training_rows"] == len(train)
    assert first["test_rows"] == len(test)
    assert first["inputs"]["test"]["sha256"] == hashlib.sha256(test_path.read_bytes()).hexdigest()
    assert first["saved_model_sha256"] == hashlib.sha256(
        (Path(first["saved_model"]) / "model.joblib").read_bytes()
    ).hexdigest()
    assert json.loads((first_dir / "evaluation.json").read_text()) == first
    per_workload = pd.read_csv(first_dir / "per_workload.csv")
    assert per_workload.groupby("model")["rows"].sum().eq(len(test)).all()

    test["power_watts"] += 400
    test.to_csv(test_path, index=False)
    second_dir = tmp_path / "second"
    second = evaluate_transfer(train_path, test_path, second_dir)
    changed_rows = pd.read_csv(second_dir / "predictions.csv")
    prediction_columns = [c for c in rows if c.endswith("_predicted_w")]
    np.testing.assert_allclose(rows[prediction_columns], changed_rows[prediction_columns])
    assert model.metadata.reference_profile == PowerModel.from_dir(second["saved_model"]).metadata.reference_profile
    assert first["scores"] != second["scores"]
    with pytest.raises(ValueError, match="empty output"):
        evaluate_transfer(train_path, test_path, first_dir)


@pytest.mark.parametrize("bad_target", [np.nan, np.inf, -1., 0., 2000.])
def test_transfer_rejects_invalid_holdout_targets_without_writing_bundle(tmp_path, bad_target):
    from gpu_power_pipeline.transfer import evaluate_transfer

    train, test = _telemetry_fixture("GPU-train"), _telemetry_fixture("GPU-test")
    test.loc[0, "power_watts"] = bad_target
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    outdir = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="invalid power targets"):
        evaluate_transfer(train_path, test_path, outdir)
    assert not outdir.exists()
