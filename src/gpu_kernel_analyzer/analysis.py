from __future__ import annotations

from dataclasses import dataclass

# Optimized/baseline kernel pairs compared at matching problem sizes. The first
# element is the optimized implementation, the second is the baseline.
SPEEDUP_PAIRS: tuple[tuple[str, str], ...] = (("gemm_tiled", "gemm_naive"),)


@dataclass(frozen=True)
class HeuristicResult:
    kernel: str
    likely_bottleneck: str
    explanation: str


@dataclass(frozen=True)
class SpeedupResult:
    optimized_kernel: str
    baseline_kernel: str
    problem_size: int
    block_size: int
    baseline_runtime_ms_mean: float
    optimized_runtime_ms_mean: float
    speedup_runtime: float
    baseline_GFLOPs: float
    optimized_GFLOPs: float
    gflops_ratio: float


def compute_speedups(summary_rows: list[dict[str, str]]) -> list[SpeedupResult]:
    """Pair optimized kernels against their baselines at matching problem sizes.

    Speedup is computed from measured mean runtime (``baseline / optimized``) so a
    value > 1.0 means the optimized kernel is faster. GFLOPs ratio is reported in
    parallel because, for a fixed FLOP count, it is the reciprocal relationship and
    serves as an independent sanity check.

    Only pairs that actually appear in ``summary_rows`` with a matching
    ``problem_size`` are emitted; missing pairs are skipped rather than guessed.
    """
    results: list[SpeedupResult] = []
    for optimized, baseline in SPEEDUP_PAIRS:
        opt_rows = {r["problem_size"]: r for r in summary_rows if r.get("kernel") == optimized}
        base_rows = {r["problem_size"]: r for r in summary_rows if r.get("kernel") == baseline}
        for problem_size in sorted(set(opt_rows) & set(base_rows), key=lambda s: int(s)):
            opt = opt_rows[problem_size]
            base = base_rows[problem_size]
            opt_rt = float(opt["runtime_ms_mean"])
            base_rt = float(base["runtime_ms_mean"])
            opt_gflops = float(opt["effective_GFLOPs"])
            base_gflops = float(base["effective_GFLOPs"])
            results.append(
                SpeedupResult(
                    optimized_kernel=optimized,
                    baseline_kernel=baseline,
                    problem_size=int(problem_size),
                    block_size=int(opt["block_size"]),
                    baseline_runtime_ms_mean=base_rt,
                    optimized_runtime_ms_mean=opt_rt,
                    speedup_runtime=(base_rt / opt_rt) if opt_rt > 0 else 0.0,
                    baseline_GFLOPs=base_gflops,
                    optimized_GFLOPs=opt_gflops,
                    gflops_ratio=(opt_gflops / base_gflops) if base_gflops > 0 else 0.0,
                )
            )
    return results


def classify_bottleneck(
    *,
    kernel: str,
    arithmetic_intensity: float,
    effective_bandwidth_GBps: float,
    effective_GFLOPs: float,
) -> HeuristicResult:
    if arithmetic_intensity < 3.0:
        bottleneck = "memory_bound_likely"
        explanation = (
            f"{kernel}: low arithmetic intensity ({arithmetic_intensity:.3f}) suggests memory pressure dominates."
        )
    elif effective_GFLOPs < 200.0 and arithmetic_intensity >= 3.0:
        bottleneck = "compute_efficiency_likely"
        explanation = (
            f"{kernel}: compute-heavy ratio with low effective GFLOPs ({effective_GFLOPs:.3f}) suggests compute inefficiency."
        )
    elif effective_bandwidth_GBps < 50.0:
        bottleneck = "underutilized_memory_path"
        explanation = (
            f"{kernel}: bandwidth ({effective_bandwidth_GBps:.3f} GB/s) appears low for a throughput benchmark."
        )
    else:
        bottleneck = "balanced_or_unknown"
        explanation = f"{kernel}: no strong single bottleneck signal from MVP heuristics."

    return HeuristicResult(kernel=kernel, likely_bottleneck=bottleneck, explanation=explanation)
