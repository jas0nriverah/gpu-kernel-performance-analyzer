from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import read_csv


def _to_float(value: str) -> float:
    return float(value)


def generate_basic_plots(run_dir: Path) -> list[Path]:
    summary_path = run_dir / "benchmark_summary.csv"
    rows = read_csv(summary_path)
    if not rows:
        return []

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # runtime vs size by kernel
    by_kernel: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        kernel = row["kernel"]
        pair = (_to_float(row["problem_size"]), _to_float(row["runtime_ms_mean"]))
        by_kernel.setdefault(kernel, []).append(pair)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for kernel, points in by_kernel.items():
        pts = sorted(points, key=lambda x: x[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=kernel)
    ax.set_title("Runtime vs Problem Size")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Runtime Mean (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    runtime_plot = plot_dir / "runtime_vs_size.png"
    fig.tight_layout()
    fig.savefig(runtime_plot, dpi=150)
    plt.close(fig)
    paths.append(runtime_plot)

    # throughput by kernel (GFLOPs)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for kernel, points in by_kernel.items():
        kernel_rows = [r for r in rows if r["kernel"] == kernel]
        kpoints = sorted(
            [(float(r["problem_size"]), float(r["effective_GFLOPs"])) for r in kernel_rows],
            key=lambda x: x[0],
        )
        ax.plot([p[0] for p in kpoints], [p[1] for p in kpoints], marker="o", label=kernel)
    ax.set_title("Effective GFLOPs vs Problem Size")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Effective GFLOPs")
    ax.grid(alpha=0.25)
    ax.legend()
    gflops_plot = plot_dir / "effective_gflops_vs_size.png"
    fig.tight_layout()
    fig.savefig(gflops_plot, dpi=150)
    plt.close(fig)
    paths.append(gflops_plot)

    return paths
