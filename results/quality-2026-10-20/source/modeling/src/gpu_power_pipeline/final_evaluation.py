"""Evaluate a frozen development winner on an untouched external device, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load_dataset
from .inference import PowerModel
from .monitoring import run_monitoring
from .transfer import baseline_predictions, regression_metrics, validate_device_holdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_capture(path: Path) -> pd.DataFrame:
    frame = load_dataset("local_nvidia_smi", data_path=str(path))
    if len(frame) != len(pd.read_csv(path)) or frame.empty:
        raise ValueError("Evaluation requires every supplied capture row; invalid targets cannot be dropped")
    if not np.isfinite(frame["power_watts"].to_numpy()).all():
        raise ValueError("Evaluation requires finite power targets")
    return frame


def validate_new_capture(training_paths: list[Path], test_path: Path, frozen_utc: str) -> None:
    """Require a later, independently identified capture, using raw provenance columns."""
    try:
        if not isinstance(frozen_utc, str):
            raise ValueError("missing freeze timestamp")
        frozen = datetime.fromisoformat(frozen_utc.replace("Z", "+00:00"))
        if frozen.tzinfo is None or frozen.utcoffset().total_seconds() != 0:
            raise ValueError("freeze timestamp must explicitly use UTC")
        frozen_epoch = frozen.timestamp()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("New capture requires a valid selection frozen_utc timestamp in UTC") from exc
    frames = [pd.read_csv(path) for path in training_paths]
    test = pd.read_csv(test_path)
    train = pd.concat(frames, ignore_index=True)
    if sha256(test_path) in {sha256(path) for path in training_paths}:
        raise ValueError("New capture cannot reuse a development input")
    for column in ["session_id", "run_id"]:
        if column == "run_id" and column not in train and column not in test:
            continue
        if column not in train or column not in test:
            raise ValueError(f"New capture requires {column} provenance")
        train_ids, test_ids = train[column].astype("string"), test[column].astype("string")
        if (
            train_ids.isna().any()
            or test_ids.isna().any()
            or train_ids.str.strip().eq("").any()
            or test_ids.str.strip().eq("").any()
        ):
            raise ValueError(f"New capture requires nonblank {column}")
        if set(train_ids.str.strip()) & set(test_ids.str.strip()):
            raise ValueError(f"New capture overlaps development {column}")

    def times(frame):
        if "epoch_s" in frame:
            values = pd.to_numeric(frame["epoch_s"], errors="coerce").to_numpy(dtype=float)
        elif "timestamp" in frame:
            parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            if parsed.isna().any():
                raise ValueError("New capture requires valid timestamps")
            values = parsed.astype("int64").to_numpy() / 1e9
        else:
            raise ValueError("New capture requires epoch_s or timestamp")
        if not np.isfinite(values).all():
            raise ValueError("New capture requires finite timestamps")
        return values

    earliest_test = min(times(test))
    if earliest_test <= frozen_epoch:
        raise ValueError("New capture must occur strictly after the selection freeze")
    if earliest_test <= max(times(train)):
        raise ValueError("New capture must occur strictly after all development rows")


def evaluate(development_dir: Path, test_path: Path, outdir: Path, scope: str = "new-device") -> dict:
    if scope not in {"new-device", "new-capture"}:
        raise ValueError("Unknown evaluation scope")
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    selection_path, plan_path = development_dir / "selection.json", development_dir / "plan.json"
    fit = json.loads((development_dir / "fit.json").read_text())
    selection = json.loads(selection_path.read_text())
    plan = json.loads(plan_path.read_text())
    if sha256(selection_path) != fit["selection_sha256"]:
        raise ValueError("Frozen selection hash mismatch")
    if sha256(plan_path) != selection["plan_sha256"]:
        raise ValueError("Frozen candidate plan hash mismatch")
    candidate = selection["candidate"]
    if candidate not in plan["candidates"] or fit["selected_candidate"] != candidate["id"]:
        raise ValueError("Frozen candidate does not match plan and fit")
    bundle = Path(fit["saved_model"])
    if not bundle.is_absolute() and not bundle.exists():
        bundle = development_dir / "registry" / candidate["id"] / bundle.name
    if sha256(bundle / "model.joblib") != fit["saved_model_sha256"]:
        raise ValueError("Saved model hash mismatch")
    train_frames = []
    for item in plan["inputs"]:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Development capture hash mismatch")
        train_frames.append(checked_capture(path))
    train = pd.concat(train_frames, ignore_index=True)
    if sorted(train["gpu_uuid"].unique().tolist()) != fit["training_gpu_uuids"] or len(train) != fit["training_rows"]:
        raise ValueError("Development capture identities or row count differ from frozen fit")
    model = PowerModel.from_dir(bundle)
    if model.feature_order != candidate["features"] or model.metadata.model_name != candidate["id"]:
        raise ValueError("Model metadata does not match frozen candidate")
    # No training or hyperparameter selection occurs after this point (or anywhere in this module).
    test = checked_capture(test_path)
    if scope == "new-device":
        validate_device_holdout(train, test)
    else:
        if test["gpu_uuid"].isna().any() or not set(test["gpu_uuid"]).issubset(set(train["gpu_uuid"])):
            raise ValueError("Known-device new capture must use a development GPU")
        validate_new_capture(
            [Path(item["path"]) for item in plan["inputs"]], test_path, selection.get("frozen_utc")
        )
    missing = set(model.feature_order) - set(test.columns)
    if missing:
        raise ValueError(f"Test capture lacks model features: {sorted(missing)}")
    with threadpool_limits(limits=1):
        predicted = model.predict(test)
    if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
        raise ValueError("Saved model returned invalid predictions")
    predictions = {candidate["id"]: predicted}
    # Names remain distinct when a selected model is itself a baseline.
    predictions.update({"pooled_" + name: values for name, values in baseline_predictions(train, test).items()})
    actual = test["power_watts"].to_numpy()
    scores = {name: regression_metrics(actual, values) for name, values in predictions.items()}
    outdir.mkdir(parents=True, exist_ok=True)
    columns = [
        name
        for name in ["timestamp", "gpu_uuid", "gpu_name", "session_id", "trial", "workload_type", "power_watts"]
        if name in test
    ]
    rows = test[columns].copy()
    for name, values in predictions.items():
        rows[name + "_predicted_w"] = values
    rows.to_csv(outdir / "predictions.csv", index=False)
    for column, filename in [("workload_type", "per_workload.csv"), ("trial", "per_trial.csv")]:
        grouped = []
        if column not in test:
            raise ValueError(f"Test capture requires {column} for grouped reporting")
        for group, indices in test.groupby(column, dropna=False).indices.items():
            for name, values in predictions.items():
                grouped.append(
                    {
                        column: group,
                        "model": name,
                        "rows": len(indices),
                        **regression_metrics(actual[indices], values[indices]),
                    }
                )
        pd.DataFrame(grouped).to_csv(outdir / filename, index=False)
    drift, _, _ = run_monitoring(model.metadata.reference_profile, test, outdir / "monitoring", input_path=test_path)
    evidence = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": scope,
        "selection_scope": selection.get("scope"),
        "selection_objective": selection.get("objective"),
        "selected_candidate": candidate["id"],
        "training_rows": len(train),
        "test_rows": len(test),
        "training_gpu_uuids": fit["training_gpu_uuids"],
        "test_gpu_uuids": sorted(test["gpu_uuid"].unique().tolist()),
        "feature_order": model.feature_order,
        "model_name": model.metadata.model_name,
        "model_version": model.metadata.version,
        "saved_model": str(bundle),
        "saved_model_sha256": fit["saved_model_sha256"],
        "selection_sha256": sha256(selection_path),
        "plan_sha256": sha256(plan_path),
        "development_inputs": plan["inputs"],
        "test_input": {"path": str(test_path.resolve()), "sha256": sha256(test_path)},
        "evaluator_sha256": sha256(Path(__file__)),
        "scores": scores,
        "drift_warnings": int(drift["severity"].eq("warning").sum()),
        "scope": "Frozen model loaded without refitting or adaptation. Evaluation scope is recorded separately; offline prediction does not change the API known-device guard. "
        "One new device/capture does not establish general portability; correlated rows and capture environments limit attribution.",
    }
    (outdir / "evaluation.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--scope", choices=["new-device", "new-capture"], default="new-device")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.development_dir, args.test, args.outdir, scope=args.scope), indent=2))


if __name__ == "__main__":
    main()
