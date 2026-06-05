from __future__ import annotations

from gpu_kernel_analyzer.analysis import compute_speedups


def _summary_row(kernel: str, problem_size: int, runtime_ms_mean: float, gflops: float) -> dict[str, str]:
    return {
        "kernel": kernel,
        "problem_size": str(problem_size),
        "block_size": "16",
        "runtime_ms_mean": str(runtime_ms_mean),
        "effective_GFLOPs": str(gflops),
    }


def test_compute_speedups_pairs_tiled_vs_naive():
    rows = [
        _summary_row("gemm_naive", 512, 1.0, 2589.0),
        _summary_row("gemm_tiled", 512, 0.5, 5178.0),
    ]
    results = compute_speedups(rows)
    assert len(results) == 1
    r = results[0]
    assert r.optimized_kernel == "gemm_tiled"
    assert r.baseline_kernel == "gemm_naive"
    assert r.problem_size == 512
    assert r.speedup_runtime == 2.0
    assert r.gflops_ratio == 2.0


def test_compute_speedups_skips_unpaired_sizes():
    rows = [
        _summary_row("gemm_naive", 256, 1.0, 100.0),
        _summary_row("gemm_tiled", 512, 0.5, 200.0),
    ]
    assert compute_speedups(rows) == []


def test_compute_speedups_handles_multiple_sizes_sorted():
    rows = [
        _summary_row("gemm_tiled", 512, 0.5, 200.0),
        _summary_row("gemm_naive", 512, 1.0, 100.0),
        _summary_row("gemm_tiled", 256, 0.4, 80.0),
        _summary_row("gemm_naive", 256, 0.8, 40.0),
    ]
    results = compute_speedups(rows)
    assert [r.problem_size for r in results] == [256, 512]
