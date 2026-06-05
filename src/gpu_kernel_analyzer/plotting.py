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


def generate_roofline_plot(
    run_dir: Path,
    *,
    peak_gflops: float | None = None,
    peak_bandwidth_GBps: float | None = None,
) -> Path | None:
    """Plot a simplified roofline: achieved (arithmetic intensity, GFLOPs) per kernel.

    Each benchmarked scenario contributes one point at its derived arithmetic
    intensity (x) and derived effective GFLOPs (y) on log-log axes.

    Ceiling lines are drawn ONLY when the caller supplies real peak numbers for the
    target GPU (peak FP32 GFLOPs and/or peak DRAM bandwidth in GB/s). Nothing about
    the hardware is assumed or invented: with no peaks provided, the plot shows the
    measured operating points alone, which still reveals memory- vs compute-bound
    clustering relative to each other.
    """
    summary_path = run_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        return None
    rows = read_csv(summary_path)
    if not rows:
        return None

    points: list[tuple[float, float, str]] = []
    for r in rows:
        try:
            ai = float(r["arithmetic_intensity"])
            gflops = float(r["effective_GFLOPs"])
        except (KeyError, ValueError):
            continue
        if ai > 0 and gflops > 0:
            points.append((ai, gflops, r["kernel"]))
    if not points:
        return None

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    kernels = sorted({p[2] for p in points})
    for kernel in kernels:
        kp = sorted([(ai, g) for ai, g, k in points if k == kernel], key=lambda x: x[0])
        ax.scatter([p[0] for p in kp], [p[1] for p in kp], label=kernel, s=40)

    ai_values = [p[0] for p in points]
    ai_min, ai_max = min(ai_values), max(ai_values)
    x_lo = ai_min / 2.0
    x_hi = ai_max * 2.0

    if peak_bandwidth_GBps and peak_bandwidth_GBps > 0:
        # Memory roof: attainable GFLOPs = peak_bandwidth (GB/s) * arithmetic_intensity.
        xs = [x_lo, x_hi]
        ys = [peak_bandwidth_GBps * x for x in xs]
        ax.plot(xs, ys, linestyle="--", color="tab:red", label=f"DRAM roof ({peak_bandwidth_GBps:.0f} GB/s)")
    if peak_gflops and peak_gflops > 0:
        ax.axhline(peak_gflops, linestyle="--", color="tab:green", label=f"Compute roof ({peak_gflops:.0f} GFLOPs)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Effective GFLOPs")
    ax.set_title("Roofline (achieved operating points)")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    output_path = plot_dir / "roofline.png"
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

    bandwidth_mem = _plot_by_kernel(
        rows,
        ["memcpy_bandwidth", "stencil_1d", "vector_add"],
        y_key="effective_bandwidth_GBps",
        title="Effective Bandwidth vs Problem Size (Memory-Bound Kernels)",
        y_label="Effective Bandwidth (GB/s)",
        output_path=plot_dir / "effective_bandwidth_vs_size_memory_kernels.png",
    )
    if bandwidth_mem:
        paths.append(bandwidth_mem)

    return paths
