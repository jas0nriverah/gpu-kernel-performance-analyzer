from __future__ import annotations

from pathlib import Path

from gpu_kernel_analyzer.artifacts import (
    PROVENANCE_COLUMNS,
    SUMMARY_COLUMNS,
    TIMING_COLUMNS,
    validate_run_directory,
    write_csv,
    write_json,
)
from gpu_kernel_analyzer.metrics import STATUS_PREDICTED, VALID_STATUSES


def test_predicted_status_is_distinct_and_excluded_from_measured_artifacts():
    assert STATUS_PREDICTED == "predicted"
    # The predicted status must never be accepted inside measured run provenance.
    assert STATUS_PREDICTED not in VALID_STATUSES


def _seed_valid_run(run_dir: Path, *, extra_provenance: list[dict] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "r1",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "benchmark_binary": "/bin/fake",
            "scenario_file": "configs/benchmark_scenarios.yaml",
            "metrics_policy_version": "mvp_v1",
        },
    )
    write_csv(
        run_dir / "timing_samples.csv",
        rows=[{"run_id": "r1", "kernel": "vector_add", "problem_size": 1024, "block_size": 128, "sample_index": 0, "runtime_ms": 0.5}],
        fieldnames=TIMING_COLUMNS,
    )
    summary_row = {col: "" for col in SUMMARY_COLUMNS}
    summary_row.update({"run_id": "r1", "kernel": "vector_add", "problem_size": 1024, "block_size": 128, "runtime_ms_mean": 0.5})
    write_csv(run_dir / "benchmark_summary.csv", rows=[summary_row], fieldnames=SUMMARY_COLUMNS)
    provenance = [
        {
            "run_id": "r1",
            "kernel": "vector_add",
            "problem_size": 1024,
            "block_size": 128,
            "metric_name": "runtime_ms",
            "metric_value": 0.5,
            "status": "measured",
            "source": "cuda_events",
        }
    ]
    if extra_provenance:
        provenance.extend(extra_provenance)
    write_csv(run_dir / "metrics_provenance.csv", rows=provenance, fieldnames=PROVENANCE_COLUMNS)


def test_valid_run_passes(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_valid_run(run_dir)
    assert validate_run_directory(run_dir).ok is True


def test_predicted_row_in_measured_provenance_is_rejected(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_valid_run(
        run_dir,
        extra_provenance=[
            {
                "run_id": "r1",
                "kernel": "vector_add",
                "problem_size": 1024,
                "block_size": 128,
                "metric_name": "runtime_ms",
                "metric_value": 0.4,
                "status": STATUS_PREDICTED,
                "source": "perf_model:numpy",
            }
        ],
    )
    report = validate_run_directory(run_dir)
    assert report.ok is False
    assert any("invalid status 'predicted'" in err for err in report.errors)
