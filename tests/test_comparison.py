from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from gpu_kernel_analyzer.comparison import Run, compare_command, compare_runs, load_measured_run


def run(value=1.0, *, name="GPU", cv=0.01, size=1024, repeats=2):
    return Run({
        "run_id": "test", "metrics_policy_version": "mvp_v1",
        "device": {"name": name, "metadata_available": True},
    }, {("vector_add", size, 128): {
        "runtime_ms_mean": str(value), "runtime_ms_median": str(value),
        "runtime_ms_cv": str(cv), "warmups": "10", "repeats": str(repeats),
    }})


def test_regression_direction_and_threshold():
    result = compare_runs(run(), run(1.2), threshold_pct=5)
    assert result["regressions"] == 1
    assert result["rows"][0]["change_pct"] == pytest.approx(20)
    assert result["rows"][0]["speedup"] == pytest.approx(1 / 1.2)
    assert not result["gate_passed"]
    assert compare_runs(run(), run(0.8))["gate_passed"]
    assert compare_runs(run(), run(1.04))["gate_passed"]
    assert compare_runs(run(), run(1.05), threshold_pct=5)["gate_passed"]


def test_device_mismatch_requires_opt_in_and_cannot_pass_gate():
    with pytest.raises(ValueError, match="Different GPUs"):
        compare_runs(run(), run(name="H100"))
    result = compare_runs(run(), run(name="H100"), allow_device_mismatch=True)
    assert result["device_mismatch"] and not result["gate_passed"]


@pytest.mark.parametrize("candidate", [run(cv=0.2), run(repeats=3)])
def test_noise_and_protocol_changes_block_gate(candidate):
    assert not compare_runs(run(), candidate)["gate_passed"]


def test_missing_coverage_is_visible():
    baseline = run()
    baseline.rows.update(run(size=2048).rows)
    result = compare_runs(baseline, run())
    assert result["missing_scenarios"] == [["vector_add", 2048, 128]]
    assert not result["gate_passed"]


def test_no_overlap_is_error():
    with pytest.raises(ValueError, match="No matching"):
        compare_runs(run(), run(size=2048))


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        compare_runs(run(), run(), threshold_pct=threshold)


def write_run(path: Path, value=1.0):
    path.mkdir()
    manifest = run().manifest
    (path / "run_manifest.json").write_text(json.dumps(manifest))
    row = dict(next(iter(run(value).rows.values())), run_id="test", kernel="vector_add",
               problem_size=1024, block_size=128, verification_passed="True", runtime_ms_cv="0")
    with (path / "benchmark_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (path / "timing_samples.csv").write_text(
        "run_id,kernel,problem_size,block_size,sample_index,runtime_ms\n"
        f"test,vector_add,1024,128,0,{value}\ntest,vector_add,1024,128,1,{value}\n"
    )


def test_load_checks_raw_samples(tmp_path):
    write_run(tmp_path / "run")
    assert load_measured_run(tmp_path / "run").rows
    timings = tmp_path / "run/timing_samples.csv"
    timings.write_text(timings.read_text().replace(",1,1.0", ",1,3.0"))
    with pytest.raises(ValueError, match="disagrees"):
        load_measured_run(tmp_path / "run")


@pytest.mark.parametrize("candidate_value, expected", [(1.0, 0), (1.2, 1)])
def test_ci_exit_code_and_reports(tmp_path, candidate_value, expected):
    write_run(tmp_path / "base")
    write_run(tmp_path / "new", candidate_value)
    args = argparse.Namespace(baseline=tmp_path / "base", candidate=tmp_path / "new",
                              outdir=tmp_path / "report", threshold_pct=5, statistic="median",
                              max_cv=0.1, allow_device_mismatch=False, fail_on_regression=True)
    assert compare_command(args) == expected
    assert (tmp_path / "report/comparison.csv").exists()
    assert json.loads((tmp_path / "report/comparison.json").read_text())["gate_passed"] == (expected == 0)
    args.outdir = args.baseline
    assert compare_command(args) == 2
