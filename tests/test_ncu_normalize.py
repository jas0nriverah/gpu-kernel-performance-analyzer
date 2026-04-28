from __future__ import annotations

from pathlib import Path

from gpu_kernel_analyzer.io import read_csv
from gpu_kernel_analyzer.ncu import normalize_ncu_raw_csv


def _write_raw_ncu_csv(path: Path, *, dram_value: str, compute_mem_value: str) -> None:
    path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process 1234 (/tmp/gpu_benchmark)",
                "==PROF== Disconnected from process 1234",
                '"ID","Kernel Name","Block Size","sm__warps_active.avg.pct_of_peak_sustained_active","sm__throughput.avg.pct_of_peak_sustained_elapsed","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","lts__throughput.avg.pct_of_peak_sustained_elapsed"',
                '"","","","","","","",""',
                f'"0","<unnamed>::vector_add_kernel(const float *)","(256, 1, 1)","76.72","21.36","{dram_value}","{compute_mem_value}","42.31"',
            ]
        ),
        encoding="utf-8",
    )


def test_normalize_raw_ncu_csv_extracts_metric_subset(tmp_path: Path):
    raw_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "normalized.csv"
    _write_raw_ncu_csv(raw_csv, dram_value="71.42", compute_mem_value="75.34")

    rows = normalize_ncu_raw_csv(
        raw_csv=raw_csv,
        normalized_csv=out_csv,
        kernel="vector_add",
        problem_size=4194304,
        block_size=256,
        metric_set="default_profiler_set",
        metric_config_path=Path(__file__).resolve().parents[1] / "configs" / "ncu_metric_sets.yaml",
    )
    assert len(rows) >= 4
    by_name = {row["metric_name"]: row["metric_value"] for row in rows}
    assert by_name["occupancy"] == "76.72"
    assert by_name["sm_utilization"] == "21.36"
    assert by_name["memory_throughput_pct"] == "71.42"
    assert by_name["l2_throughput_pct"] == "42.31"
    assert read_csv(out_csv) == rows


def test_normalize_raw_ncu_csv_uses_memory_fallback_metric(tmp_path: Path):
    raw_csv = tmp_path / "raw_fallback.csv"
    out_csv = tmp_path / "normalized_fallback.csv"
    _write_raw_ncu_csv(raw_csv, dram_value="", compute_mem_value="63.48")

    rows = normalize_ncu_raw_csv(
        raw_csv=raw_csv,
        normalized_csv=out_csv,
        kernel="vector_add",
        problem_size=4194304,
        block_size=256,
        metric_set="default_profiler_set",
        metric_config_path=Path(__file__).resolve().parents[1] / "configs" / "ncu_metric_sets.yaml",
    )
    by_name = {row["metric_name"]: row["metric_value"] for row in rows}
    assert by_name["memory_throughput_pct"] == "63.48"
