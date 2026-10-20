"""Evaluate a fixed power model on a separate GPU, without fitting on test rows.

Run with ``python -m gpu_power_pipeline.transfer --help``. The default linear
model was chosen from the published H100 study, before observing H200 outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_dataset
from .inference import PowerModel
from .monitoring import build_reference_profile, run_monitoring
from .persistence import ModelMetadata, new_version_id, save_model_bundle
from .train import fit_full_pipeline


def validate_device_holdout(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail closed when device identity is missing or crosses the split."""
    identities = []
    for label, frame in (("train", train), ("test", test)):
        if frame.empty or "gpu_uuid" not in frame:
            raise ValueError(f"{label} requires nonempty telemetry with gpu_uuid")
        ids = frame["gpu_uuid"].astype("string")
        if ids.isna().any() or ids.str.strip().eq("").any():
            raise ValueError(f"{label} has missing GPU identities")
        identities.append(set(ids.str.strip()))
    if identities[0] & identities[1]:
        raise ValueError("Training and test GPU identities overlap")


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Training-only global/workload means; unseen workloads use the global mean."""
    mean = float(train["power_watts"].mean())
    workload_means = train.groupby("workload_type")["power_watts"].mean()
    return {
        "training_mean": np.full(len(test), mean),
        "training_workload_mean": test["workload_type"].map(workload_means).fillna(mean).to_numpy(),
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation requires finite targets and predictions")
    return {
        "mae_w": float(mean_absolute_error(actual, predicted)),
        "rmse_w": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None,
    }


def evaluate_transfer(train_path: Path, test_path: Path, outdir: Path,
                      model_name: str = "linear_regression") -> dict:
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Choose an empty output directory to preserve prior evidence")
    train = load_dataset("local_nvidia_smi", data_path=str(train_path))
    test = load_dataset("local_nvidia_smi", data_path=str(test_path))
    # The general loader filters invalid targets. A holdout must never silently
    # improve its score by discarding part of the supplied capture.
    for label, path, frame in (("train", train_path, train), ("test", test_path, test)):
        if len(pd.read_csv(path)) != len(frame):
            raise ValueError(f"{label} contains invalid power targets; evaluation requires every input row")
        if not np.isfinite(frame["power_watts"].to_numpy()).all():
            raise ValueError(f"{label} requires finite power targets")
    validate_device_holdout(train, test)

    # Model choice is a caller decision. There is no search or selection using test scores.
    pipeline, numeric, categorical = fit_full_pipeline(train, model_name=model_name, random_state=42)
    features = numeric + categorical
    profile = build_reference_profile(train, features, numeric, categorical)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    metadata = ModelMetadata(
        model_name=model_name, target="power_watts", numeric_features=numeric,
        categorical_features=categorical, feature_order=features, metrics={},
        split_strategy="external_device_holdout", source="local_nvidia_smi",
        version=new_version_id(), reference_profile=profile,
        git_commit=revision.stdout.strip() or None,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = save_model_bundle(pipeline, metadata, outdir / "registry")
    model = PowerModel.from_dir(bundle)
    predicted = model.predict(test)
    np.testing.assert_allclose(predicted, pipeline.predict(test.reindex(columns=features)), rtol=1e-12, atol=1e-12)
    predictions = {model_name: predicted, **baseline_predictions(train, test)}
    actual = test["power_watts"].to_numpy()
    scores = {name: regression_metrics(actual, values) for name, values in predictions.items()}
    identifiers = [c for c in ("timestamp", "gpu_uuid", "session_id", "workload_type", "power_watts") if c in test]
    rows = test[identifiers].copy()
    for name, values in predictions.items():
        rows[f"{name}_predicted_w"] = values
    rows.to_csv(outdir / "predictions.csv", index=False)

    grouped = []
    for workload, frame in test.groupby("workload_type", dropna=False):
        indices = frame.index.to_numpy()
        for name, values in predictions.items():
            grouped.append({"workload": workload, "model": name, "rows": len(indices),
                            **regression_metrics(actual[indices], values[indices])})
    pd.DataFrame(grouped).to_csv(outdir / "per_workload.csv", index=False)
    drift, _, _ = run_monitoring(profile, test, outdir / "monitoring", input_path=test_path)
    evidence = {
        "model": model_name, "random_state": 42, "training_rows": len(train), "test_rows": len(test),
        "training_gpu_uuids": sorted(train["gpu_uuid"].unique().tolist()),
        "test_gpu_uuids": sorted(test["gpu_uuid"].unique().tolist()),
        "inputs": {label: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for label, path in (("train", train_path), ("test", test_path))},
        "feature_order": features, "scores": scores, "saved_model": str(bundle),
        "saved_model_sha256": hashlib.sha256((bundle / "model.joblib").read_bytes()).hexdigest(),
        "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": metadata.git_commit, "sklearn_version": metadata.sklearn_version,
        "drift_warnings": int(drift["severity"].eq("warning").sum()),
        "scope": "Fixed model fitted only on training GPU; test GPU used solely for evaluation. "
                 "No clipping or adaptation. Correlated telemetry rows are not independent trials. "
                 "Different capture dates and environments confound attribution to hardware alone.",
    }
    (outdir / "evaluation.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    lines = ["# External GPU power-model evaluation", "", evidence["scope"], "",
             f"Training rows: {len(train)}. Test rows: {len(test)}. Fixed model: `{model_name}`.", "",
             "| Model | MAE (W) | RMSE (W) | R² |", "| --- | ---: | ---: | ---: |"]
    for name, metrics in scores.items():
        r2 = f"{metrics['r2']:.4f}" if metrics['r2'] is not None else "undefined"
        lines.append(f"| {name} | {metrics['mae_w']:.3f} | {metrics['rmse_w']:.3f} | {r2} |")
    lines += ["", "Baselines use training targets only. Unseen workloads fall back to the training global mean.",
              "The model is saved and reloaded before scoring; transfer scores are kept separate from model metadata.",
              "", "[Evidence](evaluation.json) · [Predictions](predictions.csv) · [Per-workload errors](per_workload.csv)"]
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Training GPU's measured steady telemetry CSV")
    parser.add_argument("--test", type=Path, required=True, help="Different GPU's measured steady telemetry CSV")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", default="linear_regression",
                        choices=["linear_regression", "random_forest", "gradient_boosting"],
                        help="Fix this before inspecting test outcomes; do not tune against the test GPU")
    args = parser.parse_args()
    evidence = evaluate_transfer(args.train, args.test, args.outdir, args.model)
    print(json.dumps(evidence["scores"], indent=2))


if __name__ == "__main__":
    main()
