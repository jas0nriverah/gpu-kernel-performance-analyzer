from __future__ import annotations

import json
from pathlib import Path

from .analysis import compute_speedups
from .io import read_csv
from .schemas import PROFILER_METRICS


def _markdown_table(rows: list[dict[str, str]], columns: list[str], max_rows: int | None = 10) -> str:
    if not rows:
        return "_No rows._"
    show = rows if max_rows is None else rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in show
    ]
    return "\n".join([header, sep] + body)


def _compute_key_result(summary: list[dict[str, str]]) -> str:
    """Summarize the strongest optimized-vs-baseline speedup found in the run.

    Derived from the same ``compute_speedups`` pairing used for ``analysis_speedup.csv``
    rather than a hardcoded problem size, so it stays correct as the scenario set changes.
    """
    speedups = compute_speedups(summary)
    if not speedups:
        return "No optimized-vs-baseline kernel pair was found in this run."
    best = max(speedups, key=lambda s: s.speedup_runtime)
    return (
        f"{best.optimized_kernel} at {best.problem_size}x{best.problem_size} ran about "
        f"{best.speedup_runtime:.2f}x faster than {best.baseline_kernel} "
        f"(about {best.optimized_GFLOPs:.0f} vs {best.baseline_GFLOPs:.0f} effective GFLOPs)."
    )


def _metric_integrity_notes(provenance: list[dict[str, str]], manifest: dict[str, object]) -> list[str]:
    notes = [
        "- `runtime_ms` is measured with CUDA events.",
        "- `effective_bandwidth_GBps`, `effective_GFLOPs`, and `arithmetic_intensity` are derived estimates.",
        "- Nsight Compute timing overhead is not used for benchmark `runtime_ms` claims.",
    ]
    measured_profiler = [
        row for row in provenance
        if row.get("metric_name") in PROFILER_METRICS and row.get("status") == "measured"
    ]
    nsight = manifest.get("nsight_compute", {}) if isinstance(manifest, dict) else {}
    nsight_enabled = isinstance(nsight, dict) and bool(nsight.get("enabled")) and str(nsight.get("status")) == "parsed_metrics"
    if nsight_enabled and measured_profiler:
        notes.append("- Profiler metrics are scenario-specific and only measured for scenarios with imported Nsight CSV rows.")
    else:
        notes.append("- Profiler metrics remain unavailable unless Nsight Compute CSV metrics are imported.")
    return notes


def write_markdown_report(run_dir: Path) -> Path:
    summary = read_csv(run_dir / "benchmark_summary.csv")
    provenance = read_csv(run_dir / "metrics_provenance.csv")
    heuristics_path = run_dir / "analysis_heuristics.csv"
    heuristics = read_csv(heuristics_path) if heuristics_path.exists() else []
    speedup_path = run_dir / "analysis_speedup.csv"
    speedups = read_csv(speedup_path) if speedup_path.exists() else []
    manifest_text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    manifest_data = json.loads(manifest_text)
    key_result = _compute_key_result(summary)
    metric_notes = _metric_integrity_notes(provenance, manifest_data)

    default_metrics = [row for row in provenance if row.get("metric_name") in {
        "runtime_ms",
        "effective_bandwidth_GBps",
        "effective_GFLOPs",
        "arithmetic_intensity",
        "device_metadata",
    }]
    profiler_metrics = [row for row in provenance if row.get("metric_name") in PROFILER_METRICS]

    text = "\n".join(
        [
            "# GPU Kernel Performance Report",
            "",
            "## Run Manifest Snapshot",
            "```json",
            manifest_text.strip(),
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
                    "runtime_ms_min",
                    "runtime_ms_max",
                    "runtime_ms_cv",
                    "effective_bandwidth_GBps",
                    "effective_GFLOPs",
                    "arithmetic_intensity",
                ],
                max_rows=None,
            ),
            "",
            "## Key Result",
            key_result,
            "",
            "## Speedup vs Baseline",
            _markdown_table(
                speedups,
                [
                    "optimized_kernel",
                    "baseline_kernel",
                    "problem_size",
                    "speedup_runtime",
                    "baseline_GFLOPs",
                    "optimized_GFLOPs",
                ],
                max_rows=None,
            ),
            "",
            "## Metric Integrity Notes",
            *metric_notes,
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
                ["kernel", "problem_size", "block_size", "metric_name", "status", "metric_value", "source"],
                max_rows=None,
            ),
            "",
            "## Bottleneck Heuristics",
            _markdown_table(
                heuristics,
                ["kernel", "problem_size", "block_size", "likely_bottleneck", "explanation"],
            ),
            "",
            "## Plot Artifacts",
            "- `plots/runtime_vs_size_vector_reduction.png`",
            "- `plots/runtime_vs_size_gemm.png`",
            "- `plots/effective_gflops_vs_size_gemm.png`",
            "- `plots/effective_bandwidth_vs_size_vector_reduction.png`",
            "- `plots/effective_bandwidth_vs_size_memory_kernels.png`",
            "- `plots/roofline.png`",
        ]
    )
    out = run_dir / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    return out
