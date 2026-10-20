#!/usr/bin/env python3
"""Rebuild the archived known-device forest and verify its published predictions.

This is a retrospective numerical reproduction, not a new prospective experiment.
The original selection timestamps and held-out evidence are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

import gpu_power_pipeline.generalization as implementation
from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.generalization import CATEGORICAL, NUMERIC, build_candidate, candidate_plan
from gpu_power_pipeline.inference import PowerModel
from gpu_power_pipeline.monitoring import build_reference_profile
from gpu_power_pipeline.persistence import ModelMetadata, new_version_id, save_model_bundle
from gpu_power_pipeline.transfer import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "results/quality-2026-10-20"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def checked_frame(path: Path) -> pd.DataFrame:
    frame = load_dataset("local_nvidia_smi", data_path=str(path))
    if len(frame) != len(pd.read_csv(path)) or frame.empty or not np.isfinite(frame["power_watts"]).all():
        raise ValueError(f"Capture must retain every row and finite target: {path}")
    return frame


def reproduce(outdir: Path) -> dict:
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    archive = QUALITY / "models/known-device-development"
    plan, selection, fit = (read_json(archive / name) for name in ("plan.json", "selection.json", "fit.json"))
    if digest(archive / "plan.json") != selection["plan_sha256"]:
        raise ValueError("Archived plan hash mismatch")
    if digest(archive / "selection.json") != fit["selection_sha256"]:
        raise ValueError("Archived selection hash mismatch")
    candidate = selection["candidate"]
    if candidate not in plan["candidates"] or candidate not in candidate_plan() or candidate["id"] != fit["selected_candidate"]:
        raise ValueError("Selected configuration differs from the frozen candidate grid")
    for name, expected in plan["source_files_sha256"].items():
        if digest(Path(implementation.__file__).with_name(name)) != expected:
            raise ValueError(f"Training implementation differs from archived source: {name}")

    # Relocate by verified content, preserving the original input concatenation order.
    available = [ROOT / "results/h100-power-2026-09-22/steady_telemetry.csv",
                 QUALITY / "h200/power/steady_telemetry.csv"]
    by_hash = {digest(path): path for path in available}
    paths = [by_hash[item["sha256"]] for item in plan["inputs"]]
    training = pd.concat([checked_frame(path) for path in paths], ignore_index=True)
    if len(training) != fit["training_rows"] or sorted(training["gpu_uuid"].unique()) != fit["training_gpu_uuids"]:
        raise ValueError("Training identities or row count changed")
    features = candidate["features"]
    numeric = [name for name in features if name in NUMERIC]
    categorical = [name for name in features if name in CATEGORICAL]
    outdir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        model = build_candidate(candidate)
        model.fit(training[features], training["power_watts"])
        metadata = ModelMetadata(
            model_name=candidate["id"], target="power_watts", numeric_features=numeric,
            categorical_features=categorical, feature_order=features, metrics={},
            split_strategy="retrospective_reproduction_of_archived_selection", source="local_nvidia_smi",
            version=new_version_id(), reference_profile=build_reference_profile(training, features, numeric, categorical),
        )
        bundle = save_model_bundle(model, metadata, outdir / "registry")
        restored = PowerModel.from_dir(bundle)
        np.testing.assert_allclose(restored.predict(training), model.predict(training[features]), rtol=0, atol=1e-10)

        # Load held-out labels only after the fixed training procedure has finished.
        evaluations = {}
        for name, evaluation_dir in (
            ("h200-independent", "evaluation"), ("gh200", "known-device-challenge"), ("spark", "known-device-challenge")
        ):
            capture = QUALITY / name / "power/steady_telemetry.csv"
            evidence_dir = QUALITY / name / evaluation_dir
            evidence = read_json(evidence_dir / "evaluation.json")
            if digest(capture) != evidence["test_input"]["sha256"] or evidence["selection_sha256"] != fit["selection_sha256"]:
                raise ValueError(f"Archived capture/selection provenance mismatch: {name}")
            frame = checked_frame(capture)
            archived = pd.read_csv(evidence_dir / "predictions.csv")
            if len(frame) != len(archived):
                raise ValueError(f"Prediction row count mismatch: {name}")
            for column in ("gpu_uuid", "session_id", "trial", "workload_type"):
                if not np.array_equal(frame[column].astype(str), archived[column].astype(str)):
                    raise ValueError(f"Prediction row identity mismatch: {name}/{column}")
            if not np.array_equal(pd.to_datetime(frame["timestamp"], utc=True), pd.to_datetime(archived["timestamp"], utc=True)):
                raise ValueError(f"Prediction timestamps differ: {name}")
            np.testing.assert_allclose(frame["power_watts"], archived["power_watts"], rtol=0, atol=1e-10)
            predicted = restored.predict(frame)
            reference = archived[candidate["id"] + "_predicted_w"].to_numpy()
            np.testing.assert_allclose(predicted, reference, rtol=0, atol=1e-6,
                                       err_msg="Replay differs; use the archived dependency versions and inspect differences")
            scores = regression_metrics(frame["power_watts"].to_numpy(), predicted)
            for metric, value in scores.items():
                np.testing.assert_allclose(value, evidence["scores"][candidate["id"]][metric], rtol=0, atol=1e-6)
            evaluations[name] = {
                "rows": len(frame), "scores": scores,
                "max_absolute_prediction_difference_w": float(np.max(np.abs(predicted - reference))),
                "capture_sha256": digest(capture), "reference_predictions_sha256": digest(evidence_dir / "predictions.csv"),
            }
            pd.DataFrame({"actual_w": frame["power_watts"], "predicted_w": predicted}).to_csv(
                outdir / f"{name}-predictions.csv", index=False)

    receipt = {
        "scope": "Retrospective reproduction of the archived configuration, training data and predictions; not new test evidence.",
        "reproduced_utc": datetime.now(timezone.utc).isoformat(),
        "original_selection_frozen_utc": selection["frozen_utc"],
        "original_selection_sha256": digest(archive / "selection.json"),
        "training_inputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in paths],
        "training_rows": len(training), "candidate": candidate, "sklearn_version": sklearn.__version__,
        "archived_sklearn_version": plan["sklearn_version"], "saved_model": str(bundle),
        "saved_model_sha256": digest(bundle / "model.joblib"),
        "script_sha256": digest(Path(__file__)), "prediction_tolerance_w": 1e-6, "evaluations": evaluations,
    }
    (outdir / "reproduction.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/model-replay")
    args = parser.parse_args()
    print(json.dumps(reproduce(args.outdir), indent=2))


if __name__ == "__main__":
    main()
