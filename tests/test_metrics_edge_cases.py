from __future__ import annotations

import pytest

from gpu_kernel_analyzer.metrics import compute_derived_metrics, summarize_runtime


def test_zero_flops_memory_kernel_has_zero_gflops_and_intensity():
    # memcpy_bandwidth-style kernel: pure copy, no FLOPs.
    derived = compute_derived_metrics(bytes_moved=8_388_608, flops=0.0, runtime_ms_mean=0.05)
    assert derived["effective_GFLOPs"] == 0.0
    assert derived["arithmetic_intensity"] == 0.0
    assert derived["effective_bandwidth_GBps"] > 0.0


def test_zero_bytes_moved_yields_zero_intensity_without_error():
    derived = compute_derived_metrics(bytes_moved=0, flops=10.0, runtime_ms_mean=1.0)
    assert derived["arithmetic_intensity"] == 0.0
    assert derived["effective_bandwidth_GBps"] == 0.0


@pytest.mark.parametrize("runtime", [0.0, -1.0])
def test_non_positive_runtime_raises(runtime: float):
    with pytest.raises(ValueError):
        compute_derived_metrics(bytes_moved=1024, flops=1.0, runtime_ms_mean=runtime)


def test_summarize_runtime_empty_raises():
    with pytest.raises(ValueError):
        summarize_runtime([])


def test_summarize_runtime_single_sample_has_zero_spread():
    stats = summarize_runtime([0.5])
    assert stats.runtime_ms_min == stats.runtime_ms_max == 0.5
    assert stats.runtime_ms_stddev == 0.0
    assert stats.runtime_ms_cv == 0.0
