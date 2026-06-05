"""Performance prediction model and autotuning helpers.

This trains a small, transparent regression model on measured benchmark summaries to
predict kernel runtime and effective bandwidth from a configuration (kernel,
problem_size, block_size). It is used for two things:

* prediction: estimate runtime/bandwidth for a config without running it, and
* autotuning: rank candidate block sizes by predicted runtime.

Design notes:

* The model is a ridge regression in log10 space with a per-kernel one-hot encoding
  plus log10(problem_size) and log10(block_size). A linear model is deliberate: with
  only a handful of measured rows it is more honest and inspectable than a tree
  ensemble, and its coefficients serialize cleanly to JSON.
* scikit-learn is used for the fit when installed (the optional ``[ml]`` extra),
  otherwise a NumPy closed-form ridge solve is used. Both produce identical JSON.
* Model outputs are tagged with the ``predicted`` provenance status and written to a
  separate artifact. They are never mixed into measured benchmark artifacts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_csv
from .metrics import STATUS_PREDICTED

TRAIN_TARGETS: tuple[str, ...] = ("runtime_ms_mean", "effective_bandwidth_GBps")
PREDICTION_COLUMNS = [
    "kernel",
    "problem_size",
    "block_size",
    "metric_name",
    "metric_value",
    "status",
    "source",
]

# Below this many training rows, leave-one-out cross-validation is reported but flagged
# as not statistically meaningful.
MIN_ROWS_FOR_CV = 8
RIDGE_ALPHA = 0.1

try:  # optional dependency, provided by the [ml] extra
    from sklearn.linear_model import Ridge as _SkRidge

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn is absent
    _HAVE_SKLEARN = False


@dataclass(frozen=True)
class TrainingRow:
    kernel: str
    problem_size: int
    block_size: int
    targets: dict[str, float]


@dataclass
class PerfModel:
    backend: str
    kernel_vocab: list[str]
    feature_names: list[str]
    targets: list[str]
    coefficients: dict[str, list[float]]
    target_transform: str
    training_rows: int
    block_sizes_per_kernel: dict[str, list[int]]

    def _featurize(self, kernel: str, problem_size: int, block_size: int) -> np.ndarray:
        if kernel not in self.kernel_vocab:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Model was trained on: {', '.join(self.kernel_vocab)}."
            )
        vec = [1.0 if kernel == k else 0.0 for k in self.kernel_vocab]
        vec.append(math.log10(problem_size))
        vec.append(math.log10(block_size))
        return np.asarray(vec, dtype=float)

    def predict(self, kernel: str, problem_size: int, block_size: int) -> dict[str, float]:
        x = self._featurize(kernel, problem_size, block_size)
        out: dict[str, float] = {}
        for target in self.targets:
            log_value = float(np.dot(x, np.asarray(self.coefficients[target], dtype=float)))
            out[target] = 10.0 ** log_value
        return out

    def kernel_has_block_variation(self, kernel: str) -> bool:
        return len(self.block_sizes_per_kernel.get(kernel, [])) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "ridge_log_linear",
            "backend": self.backend,
            "kernel_vocab": self.kernel_vocab,
            "feature_names": self.feature_names,
            "targets": self.targets,
            "coefficients": self.coefficients,
            "target_transform": self.target_transform,
            "training_rows": self.training_rows,
            "block_sizes_per_kernel": self.block_sizes_per_kernel,
            "ridge_alpha": RIDGE_ALPHA,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PerfModel:
        return cls(
            backend=payload["backend"],
            kernel_vocab=list(payload["kernel_vocab"]),
            feature_names=list(payload["feature_names"]),
            targets=list(payload["targets"]),
            coefficients={k: list(v) for k, v in payload["coefficients"].items()},
            target_transform=payload["target_transform"],
            training_rows=int(payload["training_rows"]),
            block_sizes_per_kernel={k: list(v) for k, v in payload["block_sizes_per_kernel"].items()},
        )


def load_training_rows(run_dirs: list[Path]) -> list[TrainingRow]:
    """Pool benchmark_summary.csv rows from one or more run directories."""
    rows: list[TrainingRow] = []
    for run_dir in run_dirs:
        summary = run_dir / "benchmark_summary.csv"
        if not summary.exists():
            raise FileNotFoundError(f"No benchmark_summary.csv in run directory: {run_dir}")
        for r in read_csv(summary):
            targets: dict[str, float] = {}
            for target in TRAIN_TARGETS:
                try:
                    value = float(r[target])
                except (KeyError, ValueError):
                    continue
                if value > 0.0:
                    targets[target] = value
            if not targets:
                continue
            rows.append(
                TrainingRow(
                    kernel=r["kernel"],
                    problem_size=int(r["problem_size"]),
                    block_size=int(r["block_size"]),
                    targets=targets,
                )
            )
    return rows


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, str]:
    """Fit ridge coefficients without an explicit intercept (one-hot covers it)."""
    if _HAVE_SKLEARN:
        model = _SkRidge(alpha=alpha, fit_intercept=False)
        model.fit(x, y)
        return np.asarray(model.coef_, dtype=float), "sklearn"
    n_features = x.shape[1]
    gram = x.T @ x + alpha * np.eye(n_features)
    coef = np.linalg.solve(gram, x.T @ y)
    return np.asarray(coef, dtype=float), "numpy"


def train_perf_model(rows: list[TrainingRow]) -> PerfModel:
    if not rows:
        raise ValueError("Cannot train a performance model with zero rows.")
    kernel_vocab = sorted({r.kernel for r in rows})
    feature_names = [f"is_{k}" for k in kernel_vocab] + ["log10_problem_size", "log10_block_size"]

    def featurize(row: TrainingRow) -> list[float]:
        vec = [1.0 if row.kernel == k else 0.0 for k in kernel_vocab]
        vec.append(math.log10(row.problem_size))
        vec.append(math.log10(row.block_size))
        return vec

    coefficients: dict[str, list[float]] = {}
    backend = "numpy"
    trained_targets: list[str] = []
    for target in TRAIN_TARGETS:
        target_rows = [r for r in rows if target in r.targets]
        if len(target_rows) < 2:
            continue
        x = np.asarray([featurize(r) for r in target_rows], dtype=float)
        y = np.asarray([math.log10(r.targets[target]) for r in target_rows], dtype=float)
        coef, backend = _fit_ridge(x, y, RIDGE_ALPHA)
        coefficients[target] = coef.tolist()
        trained_targets.append(target)

    if not trained_targets:
        raise ValueError("Not enough rows to fit any target (need >= 2 rows per target).")

    block_sizes_per_kernel: dict[str, list[int]] = {}
    for kernel in kernel_vocab:
        sizes = sorted({r.block_size for r in rows if r.kernel == kernel})
        block_sizes_per_kernel[kernel] = sizes

    return PerfModel(
        backend=backend,
        kernel_vocab=kernel_vocab,
        feature_names=feature_names,
        targets=trained_targets,
        coefficients=coefficients,
        target_transform="log10",
        training_rows=len(rows),
        block_sizes_per_kernel=block_sizes_per_kernel,
    )


def cross_validate(rows: list[TrainingRow]) -> dict[str, Any]:
    """Leave-one-out cross-validation per target.

    Returns a dict with per-target MAE and R2 plus an ``enough_data`` flag. Metrics are
    computed even for tiny datasets, but ``enough_data`` is False below MIN_ROWS_FOR_CV so
    callers can warn that the numbers are not statistically meaningful.
    """
    result: dict[str, Any] = {"n_rows": len(rows), "enough_data": len(rows) >= MIN_ROWS_FOR_CV, "targets": {}}
    for target in TRAIN_TARGETS:
        target_rows = [r for r in rows if target in r.targets]
        if len(target_rows) < 3:
            continue
        kernel_vocab = sorted({r.kernel for r in target_rows})

        def featurize(row: TrainingRow, vocab: list[str] = kernel_vocab) -> list[float]:
            vec = [1.0 if row.kernel == k else 0.0 for k in vocab]
            vec.append(math.log10(row.problem_size))
            vec.append(math.log10(row.block_size))
            return vec

        actuals: list[float] = []
        predictions: list[float] = []
        for holdout in range(len(target_rows)):
            train_rows = [r for i, r in enumerate(target_rows) if i != holdout]
            test_row = target_rows[holdout]
            x = np.asarray([featurize(r) for r in train_rows], dtype=float)
            y = np.asarray([math.log10(r.targets[target]) for r in train_rows], dtype=float)
            coef, _ = _fit_ridge(x, y, RIDGE_ALPHA)
            log_pred = float(np.dot(np.asarray(featurize(test_row), dtype=float), coef))
            predictions.append(10.0 ** log_pred)
            actuals.append(test_row.targets[target])

        actual_arr = np.asarray(actuals, dtype=float)
        pred_arr = np.asarray(predictions, dtype=float)
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))
        ss_res = float(np.sum((actual_arr - pred_arr) ** 2))
        ss_tot = float(np.sum((actual_arr - np.mean(actual_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        result["targets"][target] = {"mae": mae, "r2": r2, "n": len(target_rows)}
    return result


@dataclass(frozen=True)
class BlockRecommendation:
    kernel: str
    problem_size: int
    recommended_block_size: int
    predicted_runtime_ms: float
    candidates: list[tuple[int, float]]
    data_backed: bool


def recommend_block_size(
    model: PerfModel, kernel: str, problem_size: int, candidates: list[int]
) -> BlockRecommendation:
    if not candidates:
        raise ValueError("At least one candidate block size is required.")
    scored = [(bs, model.predict(kernel, problem_size, bs)["runtime_ms_mean"]) for bs in candidates]
    scored_sorted = sorted(scored, key=lambda item: item[1])
    best_bs, best_runtime = scored_sorted[0]
    return BlockRecommendation(
        kernel=kernel,
        problem_size=problem_size,
        recommended_block_size=best_bs,
        predicted_runtime_ms=best_runtime,
        candidates=scored,
        data_backed=model.kernel_has_block_variation(kernel),
    )


def prediction_rows(
    kernel: str, problem_size: int, block_size: int, predicted: dict[str, float], model_source: str
) -> list[dict[str, Any]]:
    """Build model_predictions.csv rows tagged with the predicted provenance status."""
    return [
        {
            "kernel": kernel,
            "problem_size": problem_size,
            "block_size": block_size,
            "metric_name": metric_name,
            "metric_value": value,
            "status": STATUS_PREDICTED,
            "source": model_source,
        }
        for metric_name, value in predicted.items()
    ]
