import numpy as np

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.train import optional_model_status, train_and_evaluate


def test_training_smoke():
    df = load_dataset(source="synthetic", n_samples=600, random_state=21)
    results = train_and_evaluate(df, test_size=0.25, random_state=21, include_mlp=False)
    assert {"linear_regression", "random_forest", "gradient_boosting"}.issubset(results.keys())
    for _, artifacts in results.items():
        assert np.isfinite(artifacts.metrics["mae"])
        assert np.isfinite(artifacts.metrics["rmse"])
        assert np.isfinite(artifacts.metrics["r2"])
        assert np.isfinite(artifacts.residual_diagnostics["residual_std"])



def test_grouped_split_smoke():
    df = load_dataset(source="synthetic", n_samples=500, random_state=5)
    results = train_and_evaluate(df, split_strategy="grouped", random_state=5)
    assert len(results) >= 1


def test_optional_model_status_has_expected_keys():
    status = optional_model_status()
    assert {"sklearn_mlp", "torch_mlp", "xgboost", "lightgbm"}.issubset(status)
    assert all(isinstance(v, bool) for v in status.values())
