from __future__ import annotations

from pathlib import Path

from gpu_kernel_analyzer.artifacts import SUMMARY_COLUMNS, write_csv
from gpu_kernel_analyzer.plotting import generate_roofline_plot


def _seed_summary(run_dir: Path) -> None:
    rows = [
        {
            "run_id": "r1",
            "kernel": "memcpy_bandwidth",
            "problem_size": 1048576,
            "block_size": 256,
            "warmups": 10,
            "repeats": 30,
            "verification_passed": True,
            "runtime_ms_mean": 0.05,
            "runtime_ms_median": 0.05,
            "runtime_ms_min": 0.049,
            "runtime_ms_max": 0.051,
            "runtime_ms_p95": 0.051,
            "runtime_ms_stddev": 0.001,
            "runtime_ms_cv": 0.02,
            "effective_bandwidth_GBps": 1300.0,
            "effective_GFLOPs": 0.0,
            "arithmetic_intensity": 0.0,
        },
        {
            "run_id": "r1",
            "kernel": "gemm_tiled",
            "problem_size": 512,
            "block_size": 16,
            "warmups": 5,
            "repeats": 15,
            "verification_passed": True,
            "runtime_ms_mean": 0.07,
            "runtime_ms_median": 0.07,
            "runtime_ms_min": 0.069,
            "runtime_ms_max": 0.072,
            "runtime_ms_p95": 0.072,
            "runtime_ms_stddev": 0.001,
            "runtime_ms_cv": 0.01,
            "effective_bandwidth_GBps": 45.0,
            "effective_GFLOPs": 3840.0,
            "arithmetic_intensity": 85.3,
        },
    ]
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)


def test_roofline_plot_without_ceilings(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_summary(run_dir)
    path = generate_roofline_plot(run_dir)
    assert path is not None and path.exists()
    assert path.name == "roofline.png"


def test_roofline_plot_with_real_ceilings(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_summary(run_dir)
    path = generate_roofline_plot(run_dir, peak_gflops=19500.0, peak_bandwidth_GBps=1935.0)
    assert path is not None and path.exists()


def test_roofline_plot_returns_none_without_summary(tmp_path: Path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    assert generate_roofline_plot(run_dir) is None
