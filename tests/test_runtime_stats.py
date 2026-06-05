from __future__ import annotations

import math

import pytest

from gpu_kernel_analyzer.metrics import summarize_runtime


def test_summarize_runtime_includes_min_max():
    stats = summarize_runtime([0.5, 0.1, 0.3, 0.2, 0.4])
    assert stats.runtime_ms_min == pytest.approx(0.1)
    assert stats.runtime_ms_max == pytest.approx(0.5)
    assert stats.runtime_ms_min <= stats.runtime_ms_mean <= stats.runtime_ms_max
    assert stats.runtime_ms_median == pytest.approx(0.3)


def test_summarize_runtime_single_sample_has_zero_spread():
    stats = summarize_runtime([0.7])
    assert stats.runtime_ms_min == pytest.approx(0.7)
    assert stats.runtime_ms_max == pytest.approx(0.7)
    assert stats.runtime_ms_stddev == pytest.approx(0.0)
    assert stats.runtime_ms_cv == pytest.approx(0.0)


def test_summarize_runtime_cv_is_positive_for_variable_samples():
    stats = summarize_runtime([1.0, 1.1, 0.9])
    assert stats.runtime_ms_cv > 0.0
    assert math.isfinite(stats.runtime_ms_cv)


def test_summarize_runtime_rejects_empty():
    with pytest.raises(ValueError):
        summarize_runtime([])
