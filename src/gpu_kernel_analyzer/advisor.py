"""Deterministic tuning advisor.

Reads a completed run's artifacts and produces plain-language recommendations from the
measured and derived numbers: timing stability, correctness, bottleneck classification,
the tiled-vs-naive GEMM speedup, and the fastest measured block size per kernel.

It is rule-based and offline by design, so it is reproducible and safe to run in CI. An
optional LLM summary hook exists but is a stub: no LLM dependency or API key is required
for normal use or tests.
"""
from __future__ import annotations

import json
from pathlib import Path

from .io import read_csv

# Coefficient-of-variation threshold above which a measurement is flagged as noisy.
CV_UNSTABLE_THRESHOLD = 0.10


def _read_optional(run_dir: Path, name: str) -> list[dict[str, str]]:
    path = run_dir / name
    return read_csv(path) if path.exists() else []


def build_recommendations(run_dir: Path) -> list[str]:
    summary = _read_optional(run_dir, "benchmark_summary.csv")
    heuristics = _read_optional(run_dir, "analysis_heuristics.csv")
    speedups = _read_optional(run_dir, "analysis_speedup.csv")
    recs: list[str] = []

    if not summary:
        return ["No benchmark_summary.csv found. Run a sweep before requesting advice."]

    # Correctness first: a failed verification invalidates any timing discussion.
    failed = [r for r in summary if str(r.get("verification_passed", "True")).lower() == "false"]
    for r in failed:
        recs.append(
            f"[correctness] {r['kernel']} size={r['problem_size']} block={r['block_size']} "
            "failed verification. Fix correctness before trusting its timing."
        )

    # Measurement stability.
    for r in summary:
        try:
            cv = float(r.get("runtime_ms_cv", "0"))
        except ValueError:
            continue
        if cv > CV_UNSTABLE_THRESHOLD:
            recs.append(
                f"[stability] {r['kernel']} size={r['problem_size']} block={r['block_size']} "
                f"has CV={cv:.3f} (>{CV_UNSTABLE_THRESHOLD:.2f}). Increase warmups/repeats for a steadier mean."
            )

    # Bottleneck-driven hints from the heuristics file.
    for r in heuristics:
        bottleneck = r.get("likely_bottleneck", "")
        tag = f"{r.get('kernel')} size={r.get('problem_size')} block={r.get('block_size')}"
        if bottleneck == "memory_bound_likely":
            recs.append(
                f"[bottleneck] {tag}: memory-bound. Focus on coalesced access and bytes moved per launch; "
                "compute optimizations will not help much."
            )
        elif bottleneck == "compute_efficiency_likely":
            recs.append(
                f"[bottleneck] {tag}: compute-bound with low effective GFLOPs. Look at tiling, register/shared "
                "memory reuse, and occupancy."
            )
        elif bottleneck == "underutilized_memory_path":
            recs.append(
                f"[bottleneck] {tag}: low bandwidth for a throughput kernel. Check launch configuration and "
                "memory access pattern."
            )

    # Optimization payoff from the speedup pairing.
    for r in speedups:
        try:
            speedup = float(r.get("speedup_runtime", "0"))
        except ValueError:
            continue
        pair = f"{r.get('optimized_kernel')} vs {r.get('baseline_kernel')} size={r.get('problem_size')}"
        if speedup >= 1.1:
            recs.append(f"[speedup] {pair}: {speedup:.2f}x faster. The optimization is paying off.")
        else:
            recs.append(
                f"[speedup] {pair}: only {speedup:.2f}x. The optimization is not helping here; "
                "re-check tile size and problem size."
            )

    # Data-backed block-size suggestion (measured, not predicted).
    by_kernel: dict[str, list[dict[str, str]]] = {}
    for r in summary:
        by_kernel.setdefault(r["kernel"], []).append(r)
    for kernel, rows in sorted(by_kernel.items()):
        block_sizes = {r["block_size"] for r in rows}
        if len(block_sizes) < 2:
            continue
        fastest = min(rows, key=lambda r: float(r["runtime_ms_mean"]))
        recs.append(
            f"[autotune] {kernel}: among measured block sizes {sorted(int(b) for b in block_sizes)}, "
            f"block={fastest['block_size']} was fastest at size={fastest['problem_size']} "
            f"({float(fastest['runtime_ms_mean']):.4f} ms mean)."
        )

    # Roofline ceiling availability.
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        device = manifest.get("device", {}) if isinstance(manifest, dict) else {}
        if not device or not device.get("metadata_available", False):
            recs.append(
                "[roofline] No device metadata in this run. Supply --peak-gflops and --peak-bandwidth-gbps "
                "to draw real roofline ceilings."
            )

    if not recs:
        recs.append("No issues flagged. Measurements look stable and no bottleneck rule triggered.")
    return recs


def maybe_llm_summary(recommendations: list[str]) -> str | None:
    """Optional LLM summary hook.

    This is a stub: there is no bundled LLM backend and no API key is required. It always
    returns None so callers fall back to the deterministic recommendations. Wire a real
    backend here if desired without changing the rest of the workflow.
    """
    return None


def write_advisor_report(run_dir: Path, recommendations: list[str]) -> Path:
    lines = ["# Tuning Advisor", "", "Rule-based recommendations from this run's measured artifacts.", ""]
    lines.extend(f"- {rec}" for rec in recommendations)
    out = run_dir / "advisor_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
