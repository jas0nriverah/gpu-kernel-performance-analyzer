from pathlib import Path

from gpu_kernel_analyzer.artifacts import (
    PROVENANCE_COLUMNS,
    SUMMARY_COLUMNS,
    TIMING_COLUMNS,
    read_csv_rows,
    validate_run_directory,
    write_csv,
    write_json,
)
from gpu_kernel_analyzer.ncu import import_ncu_metrics


def _seed_run_dir(run_dir: Path) -> None:
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "r1",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "benchmark_binary": "fake",
            "scenario_file": "fake",
            "metrics_policy_version": "mvp_v1",
            "nsight_compute": {"enabled": False, "status": "not_detected"},
        },
    )
    write_csv(
        run_dir / "timing_samples.csv",
        rows=[
            {
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
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
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "warmups": 1,
                "repeats": 1,
                "verification_passed": True,
                "runtime_ms_mean": 0.1,
                "runtime_ms_median": 0.1,
                "runtime_ms_p95": 0.1,
                "runtime_ms_stddev": 0.0,
                "runtime_ms_cv": 0.0,
                "effective_bandwidth_GBps": 1.0,
                "effective_GFLOPs": 1.0,
                "arithmetic_intensity": 1.0,
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
                "metric_name": "occupancy",
                "metric_value": "",
                "status": "unavailable",
                "source": "nsight_compute_not_run",
            },
            {
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "sm_utilization",
                "metric_value": "",
                "status": "unavailable",
                "source": "nsight_compute_not_run",
            },
        ],
        fieldnames=PROVENANCE_COLUMNS,
    )


def test_ncu_import_exact_scenario_match(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_run_dir(run_dir)
    ncu_csv = tmp_path / "ncu.csv"
    write_csv(
        ncu_csv,
        rows=[
            {
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "occupancy",
                "metric_value": 0.62,
            }
        ],
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )

    imported = import_ncu_metrics(
        run_dir=run_dir,
        ncu_csv=ncu_csv,
        source_tool="ncu",
        source_file="reports/ncu_report.csv",
        metric_set="default_profiler_set",
    )
    assert imported == 1

    rows = read_csv_rows(run_dir / "metrics_provenance.csv")
    occupancy = [r for r in rows if r["metric_name"] == "occupancy"][0]
    assert occupancy["status"] == "measured"
    assert "source_tool=ncu" in occupancy["source"]
    assert "source_file=reports/ncu_report.csv" in occupancy["source"]
    assert "metric_set=default_profiler_set" in occupancy["source"]
    assert "import_timestamp=" in occupancy["source"]

    report = validate_run_directory(run_dir)
    assert report.ok is True


def test_ncu_import_rejects_missing_scenario_fields(tmp_path: Path):
    run_dir = tmp_path / "run_missing_fields"
    _seed_run_dir(run_dir)
    ncu_csv = tmp_path / "ncu_missing_fields.csv"
    write_csv(
        ncu_csv,
        rows=[{"kernel": "vector_add", "metric_name": "occupancy", "metric_value": 0.62}],
        fieldnames=["kernel", "metric_name", "metric_value"],
    )
    try:
        import_ncu_metrics(
            run_dir=run_dir,
            ncu_csv=ncu_csv,
            source_tool="ncu",
            source_file="reports/file.csv",
            metric_set="default",
        )
        assert False, "Expected ValueError for missing scenario fields."
    except ValueError as exc:
        assert "must contain columns" in str(exc)


def test_ncu_import_rejects_ambiguous_match(tmp_path: Path):
    run_dir = tmp_path / "run_ambiguous"
    _seed_run_dir(run_dir)
    rows = read_csv_rows(run_dir / "metrics_provenance.csv")
    rows.append(dict(rows[0]))
    write_csv(run_dir / "metrics_provenance.csv", rows=rows, fieldnames=PROVENANCE_COLUMNS)

    ncu_csv = tmp_path / "ncu_ambiguous.csv"
    write_csv(
        ncu_csv,
        rows=[
            {
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "occupancy",
                "metric_value": 0.7,
            }
        ],
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )
    try:
        import_ncu_metrics(
            run_dir=run_dir,
            ncu_csv=ncu_csv,
            source_tool="ncu",
            source_file="reports/file.csv",
            metric_set="default",
        )
        assert False, "Expected RuntimeError for ambiguous match."
    except RuntimeError as exc:
        assert "Ambiguous Nsight metric mapping" in str(exc)


def test_ncu_import_rejects_no_match(tmp_path: Path):
    run_dir = tmp_path / "run_no_match"
    _seed_run_dir(run_dir)
    ncu_csv = tmp_path / "ncu_no_match.csv"
    write_csv(
        ncu_csv,
        rows=[
            {
                "kernel": "vector_add",
                "problem_size": 9999,
                "block_size": 128,
                "metric_name": "occupancy",
                "metric_value": 0.3,
            }
        ],
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )
    try:
        import_ncu_metrics(
            run_dir=run_dir,
            ncu_csv=ncu_csv,
            source_tool="ncu",
            source_file="reports/file.csv",
            metric_set="default",
        )
        assert False, "Expected RuntimeError for no scenario match."
    except RuntimeError as exc:
        assert "No matching benchmark scenario" in str(exc)


def test_ncu_import_rejects_invalid_provenance(tmp_path: Path):
    run_dir = tmp_path / "run_bad_provenance"
    _seed_run_dir(run_dir)
    ncu_csv = tmp_path / "ncu_valid_row.csv"
    write_csv(
        ncu_csv,
        rows=[
            {
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "occupancy",
                "metric_value": 0.5,
            }
        ],
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )
    try:
        import_ncu_metrics(
            run_dir=run_dir,
            ncu_csv=ncu_csv,
            source_tool="csv",
            source_file="reports/file.csv",
            metric_set="default",
        )
        assert False, "Expected ValueError for invalid source_tool."
    except ValueError as exc:
        assert "Unsupported source_tool" in str(exc)
