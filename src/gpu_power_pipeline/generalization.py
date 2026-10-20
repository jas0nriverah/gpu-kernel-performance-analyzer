"""Bounded two-device model development; an external GPU is never an input here."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .data import load_dataset
from .inference import PowerModel
from .monitoring import build_reference_profile
from .persistence import ModelMetadata, new_version_id, save_model_bundle
from .preprocessing import build_preprocessor
from .transfer import regression_metrics, validate_device_holdout

NUMERIC = ["gpu_utilization_pct", "memory_utilization_pct", "graphics_clock_mhz", "memory_clock_mhz", "temperature_c"]
CATEGORICAL = ["workload_type", "gpu_name"]
FEATURE_SETS = {
    "full": NUMERIC + CATEGORICAL,
    "portable": [f for f in NUMERIC + CATEGORICAL if f not in {"gpu_name", "memory_clock_mhz"}],
}


class TrainingMeanRegressor(RegressorMixin, BaseEstimator):
    """Serializable training-only mean with global fallback for unseen workloads."""

    def __init__(self, by_workload: bool = False):
        self.by_workload = by_workload

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.means_ = (
            pd.DataFrame({"workload": X["workload_type"].to_numpy(), "y": np.asarray(y)})
            .groupby("workload")["y"]
            .mean()
        )
        return self

    def predict(self, X):
        if self.by_workload:
            return X["workload_type"].map(self.means_).fillna(self.mean_).to_numpy()
        return np.full(len(X), self.mean_)


def candidate_plan() -> list[dict]:
    configs = [
        ("linear", "linear_regression", {}),
        ("rf_depth6", "random_forest", {"n_estimators": 64, "max_depth": 6, "min_samples_leaf": 3}),
        ("rf_unbounded", "random_forest", {"n_estimators": 64, "max_depth": None, "min_samples_leaf": 3}),
        ("gb_lr005", "gradient_boosting", {"n_estimators": 64, "max_depth": 2, "learning_rate": 0.05}),
        ("gb_lr01", "gradient_boosting", {"n_estimators": 64, "max_depth": 2, "learning_rate": 0.1}),
    ]
    plan = [
        {
            "id": f"{name}_{ablation}",
            "model": model,
            "params": params,
            "feature_set": ablation,
            "features": list(features),
        }
        for name, model, params in configs
        for ablation, features in FEATURE_SETS.items()
    ]
    plan += [
        {"id": name, "model": name, "params": {}, "feature_set": "baseline", "features": ["workload_type"]}
        for name in ["training_mean", "training_workload_mean"]
    ]
    return plan


def build_candidate(candidate: dict) -> Pipeline:
    name = candidate["model"]
    if name.startswith("training_"):
        return Pipeline([("model", TrainingMeanRegressor(by_workload=name == "training_workload_mean"))])
    features = candidate["features"]
    numeric = [f for f in features if f in NUMERIC]
    categorical = [f for f in features if f in CATEGORICAL]
    if set(features) != set(numeric + categorical):
        raise ValueError("Candidate contains a feature outside the serving allowlist")
    constructors = {
        "linear_regression": LinearRegression,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }
    params = dict(candidate["params"])
    if name != "linear_regression":
        params["random_state"] = 42
    if name == "random_forest":
        params["n_jobs"] = 1
    return Pipeline(
        [("preprocessor", build_preprocessor(numeric, categorical)), ("model", constructors[name](**params))]
    )


def device_folds(frame: pd.DataFrame):
    devices = sorted(frame["gpu_uuid"].unique())
    if len(devices) != 2:
        raise ValueError("Selection requires exactly two development devices")
    for device in devices:
        test = np.flatnonzero(frame["gpu_uuid"].eq(device))
        train = np.flatnonzero(~frame["gpu_uuid"].eq(device))
        yield str(device), train, test


def choose_candidate(plan: list[dict], scores: list[dict]) -> dict:
    """Only device-holdout metrics can affect the frozen winner; ties use plan order."""

    def score(candidate):
        folds = [row["rmse_w"] for row in scores if row["candidate"] == candidate["id"] and row["split"] == "device"]
        if len(folds) != 2 or not np.isfinite(folds).all():
            raise ValueError("Every candidate requires two finite device fold scores")
        return float(np.mean(folds))

    winner = min(plan, key=score)
    return {
        "candidate": winner,
        "mean_device_rmse_w": score(winner),
        "selection_rule": "minimum unweighted mean of two whole-device holdout RMSE values; ties use plan order",
    }


def write_frozen(path: Path, data: dict) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def develop(paths: list[Path], outdir: Path) -> dict:
    if len(paths) != 2:
        raise ValueError("Provide exactly the H100 and H200 development captures")
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Output directory must be new or empty; frozen selections cannot be overwritten")
    frames = []
    for path in paths:
        frame = load_dataset("local_nvidia_smi", data_path=str(path))
        if len(frame) != len(pd.read_csv(path)) or not np.isfinite(frame["power_watts"]).all():
            raise ValueError("Every input row must have a finite target")
        if frame.empty or frame["gpu_uuid"].nunique() != 1:
            raise ValueError("Each development capture requires exactly one GPU")
        if not set(FEATURE_SETS["full"] + ["trial"]).issubset(frame.columns):
            raise ValueError("Development input is missing required features or trial metadata")
        if frame[FEATURE_SETS["full"] + ["trial"]].isna().any().any():
            raise ValueError("Development features and trial metadata cannot be missing")
        frames.append(frame)
    validate_device_holdout(*frames)
    frame = pd.concat(frames, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = candidate_plan()
    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": plan,
        "random_state": 42,
        "threads": 1,
        "selection": "mean whole-device holdout RMSE; whole-trial diagnostic does not select",
        "inputs": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_files_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in [
                "generalization.py",
                "preprocessing.py",
                "data.py",
                "transfer.py",
                "persistence.py",
                "inference.py",
            ]
        },
        "sklearn_version": sklearn.__version__,
    }
    write_frozen(outdir / "plan.json", configuration)
    scores, predictions = [], []

    def evaluate(split, fold, train, test, candidate):
        pipeline = build_candidate(candidate)
        features = candidate["features"]
        pipeline.fit(frame.iloc[train][features], frame.iloc[train]["power_watts"])
        values = pipeline.predict(frame.iloc[test][features])
        metrics = regression_metrics(frame.iloc[test]["power_watts"].to_numpy(), values)
        scores.append(
            {
                "split": split,
                "fold": fold,
                "candidate": candidate["id"],
                "train_rows": len(train),
                "test_rows": len(test),
                **metrics,
            }
        )
        for index, value in zip(test, values, strict=True):
            predictions.append(
                {
                    "split": split,
                    "fold": fold,
                    "candidate": candidate["id"],
                    "row": int(index),
                    "gpu_uuid": frame.iloc[index]["gpu_uuid"],
                    "trial": str(frame.iloc[index]["trial"]),
                    "workload_type": frame.iloc[index]["workload_type"],
                    "actual_w": float(frame.iloc[index]["power_watts"]),
                    "predicted_w": float(value),
                }
            )

    with threadpool_limits(limits=1):
        for fold, train, test in device_folds(frame):
            for candidate in plan:
                evaluate("device", fold, train, test, candidate)
        selection = choose_candidate(plan, scores)
        selection["frozen_utc"] = datetime.now(timezone.utc).isoformat()
        selection["plan_sha256"] = hashlib.sha256((outdir / "plan.json").read_bytes()).hexdigest()
        selection["scope"] = (
            "H100/H200 are development devices; only a later untouched external GPU can test selection. Rows are correlated."
        )
        write_frozen(outdir / "selection.json", selection)
        # Hold out whole trial groups on one device; other-device rows remain training data.
        groups = frame["gpu_uuid"].astype(str) + ":trial:" + frame["trial"].astype(str)
        for fold in sorted(groups.unique()):
            train, test = np.flatnonzero(groups.ne(fold)), np.flatnonzero(groups.eq(fold))
            if (
                frame.iloc[train]
                .loc[frame.iloc[train]["gpu_uuid"].eq(frame.iloc[test]["gpu_uuid"].iloc[0]), "trial"]
                .nunique()
                == 0
            ):
                continue
            for candidate in plan:
                evaluate("whole_trial_diagnostic", fold, train, test, candidate)
        candidate = selection["candidate"]
        features = candidate["features"]
        pipeline = build_candidate(candidate)
        pipeline.fit(frame[features], frame["power_watts"])
        numeric, categorical = [f for f in features if f in NUMERIC], [f for f in features if f in CATEGORICAL]
        metadata = ModelMetadata(
            model_name=candidate["id"],
            target="power_watts",
            numeric_features=numeric,
            categorical_features=categorical,
            feature_order=features,
            metrics={},
            split_strategy="two_device_development_after_frozen_selection",
            source="local_nvidia_smi",
            version=new_version_id(),
            reference_profile=build_reference_profile(
                frame,
                list(dict.fromkeys(features + ["gpu_name"])),
                numeric,
                list(dict.fromkeys(categorical + ["gpu_name"])),
            ),
        )
        bundle = save_model_bundle(pipeline, metadata, outdir / "registry")
        loaded = PowerModel.from_dir(bundle)
        np.testing.assert_allclose(loaded.predict(frame), pipeline.predict(frame[features]), rtol=1e-12, atol=1e-12)
    pd.DataFrame(scores).to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(outdir / "out_of_fold_predictions.csv", index=False)
    device_scores = pd.DataFrame(scores).query("split == 'device'")
    leaderboard = device_scores.groupby("candidate")["rmse_w"].agg(["mean", "max"]).reset_index()
    leaderboard.rename(columns={"mean": "mean_device_rmse_w", "max": "worst_device_rmse_w"}).to_csv(
        outdir / "leaderboard.csv", index=False
    )
    result = {
        "selection_sha256": hashlib.sha256((outdir / "selection.json").read_bytes()).hexdigest(),
        "saved_model": str(bundle),
        "saved_model_sha256": hashlib.sha256((bundle / "model.joblib").read_bytes()).hexdigest(),
        "training_rows": len(frame),
        "saved_reload_parity": True,
        "selected_candidate": candidate["id"],
        "training_gpu_uuids": sorted(frame["gpu_uuid"].unique().tolist()),
    }
    (outdir / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", type=Path, required=True)
    parser.add_argument("--h200", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(develop([args.h100, args.h200], args.outdir), indent=2))


if __name__ == "__main__":
    # Persist estimator classes under their importable package module, not __main__.
    from .generalization import main as package_main

    package_main()
