import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpu_power_pipeline import generalization as g
from gpu_power_pipeline.inference import PowerModel
from gpu_power_pipeline.monitoring import build_reference_profile
from gpu_power_pipeline.persistence import ModelMetadata, save_model_bundle


def test_whitelist_excludes_metadata_and_portable_device_proxies():
    forbidden = {
        "power_watts",
        "gpu_uuid",
        "trial",
        "timestamp",
        "epoch_s",
        "monotonic_s",
        "query_duration_s",
        "run_id",
        "session_id",
        "scenario_id",
        "phase",
        "source_file",
    }
    for candidate in g.candidate_plan():
        assert not set(candidate["features"]) & forbidden
        if candidate["feature_set"] == "portable":
            assert not set(candidate["features"]) & {"gpu_name", "memory_clock_mhz", "memory_total_mb"}


def test_device_folds_cover_every_row_once_without_device_overlap():
    frame = pd.DataFrame({"gpu_uuid": ["h100"] * 3 + ["h200"] * 5})
    held_out = []
    for _, train, test in g.device_folds(frame):
        assert not set(frame.iloc[train]["gpu_uuid"]) & set(frame.iloc[test]["gpu_uuid"])
        assert set(train) | set(test) == set(range(len(frame)))
        held_out.extend(test)
    assert sorted(held_out) == list(range(len(frame)))
    with pytest.raises(ValueError):
        list(g.device_folds(frame.iloc[:3]))


def test_selection_ignores_diagnostics_and_preserves_baseline_winner(tmp_path):
    plan = g.candidate_plan()
    scores = [
        {"candidate": candidate["id"], "split": "device", "rmse_w": 1 if candidate["id"] == "training_mean" else 10}
        for candidate in plan
        for _ in range(2)
    ]
    winner = g.choose_candidate(plan, scores)
    assert winner["candidate"]["id"] == "training_mean"
    scores.extend(
        {"candidate": c["id"], "split": "whole_trial_diagnostic", "rmse_w": 1e9 if c["id"] == "training_mean" else 0}
        for c in plan
    )
    assert g.choose_candidate(plan, scores) == winner
    path = tmp_path / "selection.json"
    g.write_frozen(path, winner)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        g.write_frozen(path, {"candidate": "changed"})
    assert path.read_bytes() == original


@pytest.mark.parametrize("baseline", ["training_mean", "training_workload_mean"])
def test_baselines_training_only_unseen_fallback_and_cross_process_reload(tmp_path, baseline):
    candidate = next(c for c in g.candidate_plan() if c["id"] == baseline)
    train = pd.DataFrame({"workload_type": ["a", "a", "b"], "gpu_name": ["H100"] * 3})
    test = pd.DataFrame({"workload_type": ["a", "unknown"]})
    pipeline = g.build_candidate(candidate).fit(train[candidate["features"]], [10.0, 20.0, 60.0])
    expected = [30.0, 30.0] if baseline == "training_mean" else [15.0, 30.0]
    np.testing.assert_allclose(pipeline.predict(test), expected)
    metadata = ModelMetadata(
        model_name=baseline,
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
    bundle = save_model_bundle(pipeline, metadata, tmp_path / "registry")
    np.testing.assert_allclose(PowerModel.from_dir(bundle).predict(test), expected)
    code = "from gpu_power_pipeline.inference import PowerModel; import sys,json; print(json.dumps(PowerModel.from_dir(sys.argv[1]).predict([{'workload_type':'a'},{'workload_type':'unknown'}]).tolist()))"
    result = subprocess.run([sys.executable, "-c", code, str(bundle)], check=True, text=True, capture_output=True)
    np.testing.assert_allclose(json.loads(result.stdout), expected)


def test_development_freezes_before_final_fit_and_saved_model_parity(tmp_path, monkeypatch):
    captures = []
    for device in ["h100", "h200"]:
        rows = []
        for trial in [1, 2]:
            for utilization in [30, 60, 90]:
                rows.append(
                    {
                        "gpu_uuid": device,
                        "gpu_name": device,
                        "trial": trial,
                        "gpu_utilization_pct": utilization,
                        "memory_utilization_pct": utilization / 2,
                        "graphics_clock_mhz": 1500.0,
                        "memory_clock_mhz": 3000.0,
                        "temperature_c": 40.0,
                        "workload_type": "compute",
                        "power_watts": 100 + utilization,
                    }
                )
        path = tmp_path / f"{device}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        captures.append(path)
    monkeypatch.setattr(g, "load_dataset", lambda source, data_path: pd.read_csv(data_path))
    # Exercise both a fitted preprocessor and a retained serializable baseline.
    plan = [g.candidate_plan()[1], g.candidate_plan()[-1]]
    monkeypatch.setattr(g, "candidate_plan", lambda: plan)
    original_save = g.save_model_bundle

    def save_after_freeze(pipeline, metadata, registry):
        selection = Path(registry).parent / "selection.json"
        assert json.loads(selection.read_text())["candidate"]["id"] == metadata.model_name
        assert "gpu_name" in metadata.reference_profile["categorical"]
        return original_save(pipeline, metadata, registry)

    monkeypatch.setattr(g, "save_model_bundle", save_after_freeze)
    outdir = tmp_path / "development"
    fit = g.develop(captures, outdir)
    assert fit["saved_reload_parity"]
    scores = pd.read_csv(outdir / "fold_metrics.csv")
    assert len(scores.query("split == 'device'")) == 4
    predictions = pd.read_csv(outdir / "out_of_fold_predictions.csv")
    for _, rows in predictions.query("split == 'device'").groupby("candidate"):
        assert sorted(rows["row"]) == list(range(12))
    frozen = (outdir / "selection.json").read_bytes()
    with pytest.raises(ValueError):
        g.develop(captures, outdir)
    assert (outdir / "selection.json").read_bytes() == frozen
