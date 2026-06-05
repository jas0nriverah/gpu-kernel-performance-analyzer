from __future__ import annotations

from pathlib import Path

from gpu_kernel_analyzer.advisor import build_recommendations, maybe_llm_summary, write_advisor_report
from gpu_kernel_analyzer.artifacts import SUMMARY_COLUMNS, write_csv


def _summary_row(kernel: str, problem_size: int, block_size: int, runtime: float, cv: float, verified: bool = True) -> dict:
    row = {col: "" for col in SUMMARY_COLUMNS}
    row.update(
        {
            "run_id": "r1",
            "kernel": kernel,
            "problem_size": problem_size,
            "block_size": block_size,
            "verification_passed": verified,
            "runtime_ms_mean": runtime,
            "runtime_ms_cv": cv,
            "effective_bandwidth_GBps": 900.0,
            "effective_GFLOPs": 0.0,
            "arithmetic_intensity": 0.0,
        }
    )
    return row


def test_advisor_flags_instability_and_block_choice(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _summary_row("vector_add", 1_048_576, 128, 0.30, 0.25),  # unstable: high CV
        _summary_row("vector_add", 1_048_576, 256, 0.22, 0.01),
    ]
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)

    recs = build_recommendations(run_dir)
    joined = "\n".join(recs)
    assert "[stability]" in joined  # high CV row flagged
    assert "[autotune]" in joined   # fastest measured block reported
    assert "block=256" in joined    # 256 was the faster measured block


def test_advisor_flags_failed_verification(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_summary_row("gemm_tiled", 512, 16, 0.09, 0.01, verified=False)]
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)
    recs = build_recommendations(run_dir)
    assert any("[correctness]" in r for r in recs)


def test_advisor_handles_missing_summary(tmp_path: Path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    recs = build_recommendations(run_dir)
    assert recs and "No benchmark_summary.csv" in recs[0]


def test_llm_hook_is_optional_stub():
    assert maybe_llm_summary(["anything"]) is None


def test_write_advisor_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_summary_row("vector_add", 1_048_576, 256, 0.22, 0.01)]
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)
    recs = build_recommendations(run_dir)
    path = write_advisor_report(run_dir, recs)
    assert path.exists()
    assert path.name == "advisor_report.md"
    assert "Tuning Advisor" in path.read_text(encoding="utf-8")
