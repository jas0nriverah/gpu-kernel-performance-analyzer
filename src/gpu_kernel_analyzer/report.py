from __future__ import annotations

from pathlib import Path

from .io import read_csv


def _markdown_table(rows: list[dict[str, str]], columns: list[str], max_rows: int = 10) -> str:
    if not rows:
        return "_No rows._"
    show = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in show
    ]
    return "\n".join([header, sep] + body)


def write_markdown_report(run_dir: Path) -> Path:
    summary = read_csv(run_dir / "benchmark_summary.csv")
    provenance = read_csv(run_dir / "metrics_provenance.csv")
    heuristics_path = run_dir / "analysis_heuristics.csv"
    heuristics = read_csv(heuristics_path) if heuristics_path.exists() else []
    manifest = (run_dir / "run_manifest.json").read_text(encoding="utf-8")

    default_metrics = [row for row in provenance if row.get("metric_name") in {
        "runtime_ms",
        "effective_bandwidth_GBps",
        "effective_GFLOPs",
        "arithmetic_intensity",
        "device_metadata",
    }]
    profiler_metrics = [row for row in provenance if row.get("metric_name") in {
        "occupancy",
        "sm_utilization",
        "l2_cache_hit_rate",
    }]

    text = "\n".join(
        [
            "# GPU Kernel Performance Report",
            "",
            "## Run Manifest Snapshot",
            "```json",
            manifest.strip(),
            "```",
            "",
            "## Benchmark Summary",
            _markdown_table(
                summary,
                [
                    "kernel",
                    "problem_size",
                    "block_size",
                    "runtime_ms_mean",
                    "effective_bandwidth_GBps",
                    "effective_GFLOPs",
                    "arithmetic_intensity",
                ],
            ),
            "",
            "## Metrics Provenance (Default Metrics)",
            _markdown_table(
                default_metrics,
                ["kernel", "problem_size", "block_size", "metric_name", "status", "source", "metric_value"],
            ),
            "",
            "## Profiler Metrics Status",
            _markdown_table(
                profiler_metrics,
                ["kernel", "problem_size", "block_size", "metric_name", "status", "source"],
            ),
            "",
            "## Bottleneck Heuristics",
            _markdown_table(
                heuristics,
                ["kernel", "problem_size", "block_size", "likely_bottleneck", "explanation"],
            ),
            "",
            "## Plot Artifacts",
            "- `plots/runtime_vs_size.png`",
            "- `plots/effective_gflops_vs_size.png`",
        ]
    )
    out = run_dir / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    return out
