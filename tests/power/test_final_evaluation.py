import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpu_power_pipeline import final_evaluation as final
from gpu_power_pipeline.generalization import build_candidate, candidate_plan, write_frozen
from gpu_power_pipeline.inference import PowerModel
from gpu_power_pipeline.monitoring import build_reference_profile
from gpu_power_pipeline.persistence import ModelMetadata, save_model_bundle


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "load_dataset", lambda source, data_path: pd.read_csv(data_path))
    development = tmp_path / "development"
    development.mkdir()
    train = pd.DataFrame(
        {
            "gpu_uuid": ["a", "a", "b", "b"],
            "gpu_name": ["H100", "H100", "H200", "H200"],
            "workload_type": ["compute", "memory", "compute", "memory"],
            "power_watts": [100.0, 200.0, 120.0, 220.0],
            "trial": [1, 1, 2, 2],
        }
    )
    train["session_id"] = ["dev1", "dev1", "dev2", "dev2"]
    train["epoch_s"] = [1.0, 2.0, 3.0, 4.0]
    inputs = []
    for device, rows in train.groupby("gpu_uuid"):
        path = tmp_path / f"{device}.csv"
        rows.to_csv(path, index=False)
        inputs.append({"path": str(path), "sha256": final.sha256(path)})
    candidate = candidate_plan()[-1]
    pipeline = build_candidate(candidate).fit(train[candidate["features"]], train["power_watts"])
    metadata = ModelMetadata(
        model_name=candidate["id"],
        target="power_watts",
        numeric_features=[],
        categorical_features=["workload_type"],
        feature_order=["workload_type"],
        metrics={},
        split_strategy="test",
        source="test",
        version="fixed",
        reference_profile=build_reference_profile(
            train, ["workload_type", "gpu_name"], [], ["workload_type", "gpu_name"]
        ),
    )
    bundle = save_model_bundle(pipeline, metadata, development / "registry")
    write_frozen(development / "plan.json", {"inputs": inputs, "candidates": [candidate]})
    write_frozen(
        development / "selection.json", {"candidate": candidate, "plan_sha256": final.sha256(development / "plan.json"),
         "frozen_utc": "1970-01-01T00:00:05+00:00"}
    )
    fit = {
        "selection_sha256": final.sha256(development / "selection.json"),
        "selected_candidate": candidate["id"],
        "saved_model": str(bundle),
        "saved_model_sha256": final.sha256(bundle / "model.joblib"),
        "training_gpu_uuids": ["a", "b"],
        "training_rows": len(train),
    }
    (development / "fit.json").write_text(json.dumps(fit))
    test = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "gpu_uuid": ["external", "external"],
            "gpu_name": ["external", "external"],
            "workload_type": ["compute", "new_workload"],
            "power_watts": [115.0, 165.0],
            "trial": [1, 2],
        }
    ).to_csv(test, index=False)
    return development, test, tmp_path / "evaluation", bundle


def test_frozen_prediction_parity_and_grouped_artifacts(experiment, monkeypatch):
    development, test, output, bundle = experiment
    expected = PowerModel.from_dir(bundle).predict(pd.read_csv(test))

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Final evaluation must never refit")

    from gpu_power_pipeline.generalization import TrainingMeanRegressor

    monkeypatch.setattr(TrainingMeanRegressor, "fit", forbidden_fit)
    result = final.evaluate(development, test, output)
    predictions = pd.read_csv(output / "predictions.csv")
    np.testing.assert_allclose(predictions["training_workload_mean_predicted_w"], expected)
    np.testing.assert_allclose(expected, [110.0, 160.0])
    assert result["test_rows"] == 2
    assert result["test_input"]["sha256"] == final.sha256(test)
    assert set(result["scores"]) == {"training_workload_mean", "pooled_training_mean", "pooled_training_workload_mean"}
    assert pd.read_csv(output / "per_trial.csv")["rows"].eq(1).all()
    assert (output / "monitoring/drift_report.csv").is_file()


@pytest.mark.parametrize("tamper", ["selection", "plan", "model", "training"])
def test_frozen_hash_tampering_rejected_before_test_load(experiment, monkeypatch, tamper):
    development, test, output, bundle = experiment
    training = Path(json.loads((development / "plan.json").read_text())["inputs"][0]["path"])
    target = {
        "selection": development / "selection.json",
        "plan": development / "plan.json",
        "model": bundle / "model.joblib",
        "training": training,
    }[tamper]
    with target.open("ab") as handle:
        handle.write(b" ")
    original = final.load_dataset

    def load(source, data_path):
        assert Path(data_path) != test, "External test must remain unread on invalid provenance"
        return original(source, data_path)

    monkeypatch.setattr(final, "load_dataset", load)
    with pytest.raises(ValueError, match="hash mismatch"):
        final.evaluate(development, test, output)


def test_overlapping_gpu_rejected(experiment):
    development, test, output, _ = experiment
    frame = pd.read_csv(test)
    frame["gpu_uuid"] = "a"
    frame.to_csv(test, index=False)
    with pytest.raises(ValueError, match="overlap"):
        final.evaluate(development, test, output)


def test_row_drop_rejected(experiment, monkeypatch):
    development, test, output, _ = experiment
    original = final.load_dataset
    monkeypatch.setattr(
        final,
        "load_dataset",
        lambda source, data_path: (
            original(source, data_path).iloc[:1] if Path(data_path) == test else original(source, data_path)
        ),
    )
    with pytest.raises(ValueError, match="every supplied capture row"):
        final.evaluate(development, test, output)


def test_existing_output_and_wrong_metadata_rejected(experiment):
    development, test, output, bundle = experiment
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        final.evaluate(development, test, output)
    assert sentinel.read_text() == "keep"
    metadata = bundle / "metadata.json"
    data = json.loads(metadata.read_text())
    data["feature_order"] = ["gpu_name"]
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata"):
        final.evaluate(development, test, output.parent / "fresh")


@pytest.mark.parametrize("fault", [None, "early", "session", "unseen_gpu", "missing_timestamp"])
def test_new_capture_requires_future_disjoint_session_on_known_gpu(experiment, fault):
    development, test, output, _ = experiment
    frame = pd.read_csv(test)
    frame["gpu_uuid"] = "a"
    frame["session_id"] = "fresh"
    frame["epoch_s"] = [10.0, 11.0]
    if fault == "early":
        frame["epoch_s"] = [4.0, 5.0]
    elif fault == "session":
        frame["session_id"] = "dev1"
    elif fault == "unseen_gpu":
        frame["gpu_uuid"] = "unknown"
    elif fault == "missing_timestamp":
        frame = frame.drop(columns="epoch_s")
    frame.to_csv(test, index=False)
    if fault:
        with pytest.raises(ValueError):
            final.evaluate(development, test, output, scope="new-capture")
    else:
        result = final.evaluate(development, test, output, scope="new-capture")
        assert result["evaluation_scope"] == "new-capture"
        assert result["test_gpu_uuids"] == ["a"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_external_targets_rejected(experiment, value):
    development, test, output, _ = experiment
    frame = pd.read_csv(test)
    frame.loc[0, "power_watts"] = value
    frame.to_csv(test, index=False)
    with pytest.raises(ValueError, match="finite power targets"):
        final.evaluate(development, test, output)


@pytest.mark.parametrize("frozen_utc", [None, "invalid", "1970-01-01T00:00:05", "1970-01-01T01:00:05+01:00"])
def test_new_capture_rejects_missing_or_invalid_utc_freeze(experiment, frozen_utc):
    development, test, output, _ = experiment
    selection_path = development / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["frozen_utc"] = frozen_utc
    selection_path.write_text(json.dumps(selection))
    fit_path = development / "fit.json"
    fit = json.loads(fit_path.read_text())
    fit["selection_sha256"] = final.sha256(selection_path)
    fit_path.write_text(json.dumps(fit))
    frame = pd.read_csv(test)
    frame["gpu_uuid"] = "a"
    frame["session_id"] = "fresh"
    frame["epoch_s"] = [10.0, 11.0]
    frame.to_csv(test, index=False)
    with pytest.raises(ValueError, match="valid selection frozen_utc"):
        final.evaluate(development, test, output, scope="new-capture")
    assert not output.exists()


@pytest.mark.parametrize("earliest", [4.5, 5.0])
def test_capture_after_training_but_not_after_freeze_rejected(experiment, earliest):
    development, test, output, _ = experiment
    # Training ends at epoch 4, selection freezes at 5. A different session
    # between those times (or at the freeze) is not prospective validation.
    frame = pd.read_csv(test)
    frame["gpu_uuid"] = "a"
    frame["session_id"] = "fresh"
    frame["epoch_s"] = [earliest, 6.0]
    frame.to_csv(test, index=False)
    with pytest.raises(ValueError, match="strictly after the selection freeze"):
        final.evaluate(development, test, output, scope="new-capture")
    assert not output.exists()
