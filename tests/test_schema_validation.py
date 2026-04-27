from pathlib import Path

from gpu_kernel_analyzer.artifacts import (
    PROVENANCE_COLUMNS,
    SUMMARY_COLUMNS,
    TIMING_COLUMNS,
    validate_run_directory,
    write_csv,
    write_json,
)


def _build_valid_run_dir(run_dir: Path) -> None:
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "test_run",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "benchmark_binary": "fake",
            "scenario_file": "fake",
            "metrics_policy_version": "mvp_v1",
        },
    )

    write_csv(
        run_dir / "timing_samples.csv",
        rows=[
            {
                "run_id": "test_run",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 256,
                "sample_index": 0,
                "runtime_ms": 0.1,
            }
        ],
        fieldnames=TIMING_COLUMNS,
    )

    write_csv(
        run_dir / "benchmark_summary.csv",
        rows=[
            {
                "run_id": "test_run",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 256,
                "warmups": 1,
                "repeats": 1,
                "verification_passed": True,
                "runtime_ms_mean": 0.1,
                "runtime_ms_median": 0.1,
                "runtime_ms_p95": 0.1,
                "runtime_ms_stddev": 0.0,
                "runtime_ms_cv": 0.0,
                "effective_bandwidth_GBps": 12.0,
                "effective_GFLOPs": 4.0,
                "arithmetic_intensity": 0.33,
            }
        ],
        fieldnames=SUMMARY_COLUMNS,
    )

    write_csv(
        run_dir / "metrics_provenance.csv",
        rows=[
            {
                "run_id": "test_run",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 256,
                "metric_name": "runtime_ms",
                "metric_value": 0.1,
                "status": "measured",
                "source": "cuda_events",
            },
            {
                "run_id": "test_run",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 256,
                "metric_name": "occupancy",
                "metric_value": "",
                "status": "unavailable",
                "source": "nsight_compute_not_run",
            },
        ],
        fieldnames=PROVENANCE_COLUMNS,
    )


def test_validate_run_directory_ok(tmp_path: Path):
    run_dir = tmp_path / "run_ok"
    _build_valid_run_dir(run_dir)
    report = validate_run_directory(run_dir)
    assert report.ok is True
    assert report.errors == []


def test_validate_run_directory_rejects_invalid_occupancy_status(tmp_path: Path):
    run_dir = tmp_path / "run_bad"
    _build_valid_run_dir(run_dir)
    # csv is not JSON; overwrite directly with invalid CSV content
    (run_dir / "metrics_provenance.csv").write_text(
        "run_id,kernel,problem_size,block_size,metric_name,metric_value,status,source\n"
        "test_run,vector_add,1024,256,occupancy,0.5,measured,cuda_events\n",
        encoding="utf-8",
    )
    report = validate_run_directory(run_dir)
    assert report.ok is False
    assert any("cannot be measured" in err for err in report.errors)
