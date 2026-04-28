from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import read_csv


def _to_float(value: str) -> float:
    return float(value)


def _plot_by_kernel(
    rows: list[dict[str, str]],
    kernels: list[str],
    *,
    y_key: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> Path | None:
    selected = [r for r in rows if r["kernel"] in kernels]
    if not selected:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for kernel in kernels:
        krows = [r for r in selected if r["kernel"] == kernel]
        if not krows:
            continue
        points = sorted(
            [(float(r["problem_size"]), float(r[y_key])) for r in krows],
            key=lambda x: x[0],
        )
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=kernel)
    ax.set_title(title)
    ax.set_xlabel("Problem Size")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_basic_plots(run_dir: Path) -> list[Path]:
    summary_path = run_dir / "benchmark_summary.csv"
    rows = read_csv(summary_path)
    if not rows:
        return []

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # Remove deprecated single-axis plot filenames to avoid confusion.
    for legacy_name in ("runtime_vs_size.png", "effective_gflops_vs_size.png"):
        legacy_path = plot_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    runtime_vr = _plot_by_kernel(
        rows,
        ["vector_add", "reduction"],
        y_key="runtime_ms_mean",
        title="Runtime vs Problem Size (Vector Add + Reduction)",
        y_label="Runtime Mean (ms)",
        output_path=plot_dir / "runtime_vs_size_vector_reduction.png",
    )
    if runtime_vr:
        paths.append(runtime_vr)

    runtime_gemm = _plot_by_kernel(
        rows,
        ["gemm_naive", "gemm_tiled"],
        y_key="runtime_ms_mean",
        title="Runtime vs Problem Size (GEMM)",
        y_label="Runtime Mean (ms)",
        output_path=plot_dir / "runtime_vs_size_gemm.png",
    )
    if runtime_gemm:
        paths.append(runtime_gemm)

    gflops_gemm = _plot_by_kernel(
        rows,
        ["gemm_naive", "gemm_tiled"],
        y_key="effective_GFLOPs",
        title="Effective GFLOPs vs Problem Size (GEMM)",
        y_label="Effective GFLOPs",
        output_path=plot_dir / "effective_gflops_vs_size_gemm.png",
    )
    if gflops_gemm:
        paths.append(gflops_gemm)

    bandwidth_vr = _plot_by_kernel(
        rows,
        ["vector_add", "reduction"],
        y_key="effective_bandwidth_GBps",
        title="Effective Bandwidth vs Problem Size (Vector Add + Reduction)",
        y_label="Effective Bandwidth (GB/s)",
        output_path=plot_dir / "effective_bandwidth_vs_size_vector_reduction.png",
    )
    if bandwidth_vr:
        paths.append(bandwidth_vr)

    return paths
