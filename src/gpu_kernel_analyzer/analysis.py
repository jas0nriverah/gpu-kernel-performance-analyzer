from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeuristicResult:
    kernel: str
    likely_bottleneck: str
    explanation: str


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
