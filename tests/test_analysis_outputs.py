from pathlib import Path

from gpu_kernel_analyzer.plotting import generate_basic_plots
from gpu_kernel_analyzer.report import write_markdown_report
from gpu_kernel_analyzer.artifacts import write_csv, write_json, SUMMARY_COLUMNS, PROVENANCE_COLUMNS


def _seed_run_dir(run_dir: Path) -> None:
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "r1",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "benchmark_binary": "fake",
            "scenario_file": "fake",
            "metrics_policy_version": "mvp_v1",
            "nsight_compute": {"enabled": False, "status": "not_run"},
        },
    )
    write_csv(
        run_dir / "benchmark_summary.csv",
        rows=[
            {
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "warmups": 1,
                "repeats": 3,
                "verification_passed": True,
                "runtime_ms_mean": 0.1,
                "runtime_ms_median": 0.1,
                "runtime_ms_p95": 0.11,
                "runtime_ms_stddev": 0.01,
                "runtime_ms_cv": 0.1,
                "effective_bandwidth_GBps": 20.0,
                "effective_GFLOPs": 5.0,
                "arithmetic_intensity": 0.25,
            }
        ],
        fieldnames=SUMMARY_COLUMNS,
    )
    write_csv(
        run_dir / "metrics_provenance.csv",
        rows=[
            {
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "runtime_ms",
                "metric_value": 0.1,
                "status": "measured",
                "source": "cuda_events",
            },
            {
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "occupancy",
                "metric_value": "",
                "status": "unavailable",
                "source": "nsight_compute_not_run",
            },
        ],
        fieldnames=PROVENANCE_COLUMNS,
    )
    write_csv(
        run_dir / "analysis_heuristics.csv",
        rows=[
            {
                "kernel": "vector_add",
                "problem_size": "1024",
                "block_size": "128",
                "likely_bottleneck": "memory_bound_likely",
                "explanation": "test explanation",
            }
        ],
        fieldnames=["kernel", "problem_size", "block_size", "likely_bottleneck", "explanation"],
    )


def test_generate_plots_and_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_run_dir(run_dir)
    plots = generate_basic_plots(run_dir)
    expected_plot_names = {
        "runtime_vs_size_vector_reduction.png",
        "effective_bandwidth_vs_size_vector_reduction.png",
    }
    assert expected_plot_names.issubset({p.name for p in plots})
    for path in plots:
        assert path.exists()

    report_path = write_markdown_report(run_dir)
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "GPU Kernel Performance Report" in text
    assert "Profiler Metrics Status" in text
    assert "runtime_vs_size_vector_reduction.png" in text
