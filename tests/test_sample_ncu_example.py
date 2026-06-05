from __future__ import annotations

from pathlib import Path

from gpu_kernel_analyzer.ncu import normalize_ncu_raw_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RAW = REPO_ROOT / "examples" / "ncu_raw_vector_add_4194304_b256.csv"
METRIC_CONFIG = REPO_ROOT / "configs" / "ncu_metric_sets.yaml"


def test_committed_sample_ncu_csv_normalizes(tmp_path: Path):
    assert SAMPLE_RAW.exists(), "sample Nsight raw CSV should be committed for GPU-less demos"
    out_csv = tmp_path / "normalized.csv"
    rows = normalize_ncu_raw_csv(
        raw_csv=SAMPLE_RAW,
        normalized_csv=out_csv,
        kernel="vector_add",
        problem_size=4194304,
        block_size=256,
        metric_set="default_profiler_set",
        metric_config_path=METRIC_CONFIG,
    )
    by_name = {row["metric_name"]: row["metric_value"] for row in rows}
    assert by_name["occupancy"] == "76.72"
    assert by_name["sm_utilization"] == "21.36"
    assert by_name["memory_throughput_pct"] == "71.42"
    assert by_name["l2_throughput_pct"] == "77.37"
