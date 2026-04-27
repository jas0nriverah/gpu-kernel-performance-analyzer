from gpu_kernel_analyzer.analysis import classify_bottleneck


def test_memory_bound_classification():
    result = classify_bottleneck(
        kernel="vector_add",
        arithmetic_intensity=0.2,
        effective_bandwidth_GBps=400.0,
        effective_GFLOPs=80.0,
    )
    assert result.likely_bottleneck == "memory_bound_likely"


def test_compute_efficiency_classification():
    result = classify_bottleneck(
        kernel="gemm_naive",
        arithmetic_intensity=8.0,
        effective_bandwidth_GBps=120.0,
        effective_GFLOPs=100.0,
    )
    assert result.likely_bottleneck == "compute_efficiency_likely"
