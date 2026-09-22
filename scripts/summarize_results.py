"""Regenerate publication tables and charts from the committed independent sweeps.

Usage: python scripts/summarize_results.py --results results/h100-2026-09-22
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpu_kernel_analyzer.artifacts import write_csv
from gpu_kernel_analyzer.comparison import load_measured_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_measured_run(path) for path in sorted(args.results.glob("trial-*")) if path.is_dir()]
    if len(runs) < 2:
        raise ValueError("At least two independent trial directories are required")
    keys = set(runs[0].rows)
    if any(set(run.rows) != keys or run.manifest["device"] != runs[0].manifest["device"] for run in runs):
        raise ValueError("Trial scenarios and devices must match")
    rows = []
    for key in sorted(keys):
        values = [run.rows[key] for run in runs]
        means = [float(row["runtime_ms_mean"]) for row in values]
        rows.append({
            "kernel": key[0], "problem_size": key[1], "block_size": key[2], "trials": len(runs),
            "median_mean_ms": median(means), "min_mean_ms": min(means), "max_mean_ms": max(means),
            "spread_pct": (max(means) / min(means) - 1) * 100,
            "median_bandwidth_GBps": median(float(row["effective_bandwidth_GBps"]) for row in values),
            "median_GFLOPs": median(float(row["effective_GFLOPs"]) for row in values),
            "max_within_run_cv": max(float(row["runtime_ms_cv"]) for row in values),
        })
    write_csv(args.results / "aggregate_summary.csv", rows, list(rows[0]))
    out = args.results / "charts"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc", "axes.facecolor": "#f8fafc"})
    colors = ["#0f766e", "#2563eb", "#b45309", "#7c3aed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for kernel, color in zip(["gemm_naive", "gemm_tiled"], colors, strict=False):
        selected = [r for r in rows if r["kernel"] == kernel]
        axes[0].plot([r["problem_size"] for r in selected], [r["median_GFLOPs"] / 1000 for r in selected],
                     marker="o", color=color, label=kernel.replace("gemm_", ""))
    axes[0].set(xlabel="Square matrix dimension", ylabel="Effective FP32 TFLOP/s", title="Shared-memory tiling improves GEMM")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([128, 256, 512, 1024, 2048], [128, 256, 512, 1024, 2048])
    for kernel, color in zip(["vector_add", "memcpy_bandwidth", "stencil_1d", "reduction"], colors, strict=True):
        selected = [r for r in rows if r["kernel"] == kernel and r["block_size"] == 256]
        axes[1].plot([r["problem_size"] / 2**20 for r in selected], [r["median_bandwidth_GBps"] / 1000 for r in selected],
                     marker="o", color=color, label=kernel.replace("_bandwidth", ""))
    axes[1].set(xlabel="Elements (millions, base 2)", ylabel="Effective bandwidth (TB/s)", title="Memory workloads · block size 256")
    axes[1].set_xscale("log", base=4)
    axes[1].set_xticks([1, 4, 16, 64], [1, 4, 16, 64])
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0)
    fig.suptitle("H100 80GB · median of three run means · 50 samples per scenario", fontsize=14, fontweight="bold")
    fig.savefig(out / "scaling.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for kernel, color in zip(["vector_add", "memcpy_bandwidth", "stencil_1d", "reduction"], colors, strict=True):
        selected = [r for r in rows if r["kernel"] == kernel and r["problem_size"] == 67108864]
        ax.plot([r["block_size"] for r in selected], [r["median_mean_ms"] for r in selected],
                marker="o", label=kernel.replace("_bandwidth", ""), color=color)
    ax.set(xlabel="Threads per block", ylabel="Mean kernel runtime (ms)", title="Block size matters · 67,108,864 elements")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512], [64, 128, 256, 512])
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(out / "block_sizes.png", dpi=170)
    plt.close(fig)

    lines = ["# H100 aggregate results", "", "Median of three independent run means; ranges span those means, not confidence intervals.", "",
             "| Kernel | Size | Block | Mean ms (median) | Range ms | Effective GB/s | Effective GFLOP/s | Max CV |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['kernel']} | {r['problem_size']} | {r['block_size']} | {r['median_mean_ms']:.6f} | "
                     f"{r['min_mean_ms']:.6f}–{r['max_mean_ms']:.6f} | {r['median_bandwidth_GBps']:.1f} | "
                     f"{r['median_GFLOPs']:.1f} | {r['max_within_run_cv']:.3f} |")
    (args.results / "AGGREGATE.md").write_text("\n".join(lines) + "\n")
    print(f"Aggregated {len(rows)} scenarios from {len(runs)} independent sweeps into {args.results}")


if __name__ == "__main__":
    main()
