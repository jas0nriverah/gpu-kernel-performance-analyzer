from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, median, pstdev


STATUS_MEASURED = "measured"
STATUS_DERIVED = "derived_estimate"
STATUS_UNAVAILABLE = "unavailable"
VALID_STATUSES = {STATUS_MEASURED, STATUS_DERIVED, STATUS_UNAVAILABLE}


@dataclass(frozen=True)
class SummaryStats:
    runtime_ms_mean: float
    runtime_ms_median: float
    runtime_ms_p95: float
    runtime_ms_stddev: float
    runtime_ms_cv: float


def summarize_runtime(samples_ms: list[float]) -> SummaryStats:
    if not samples_ms:
        raise ValueError("Runtime samples cannot be empty.")
    ordered = sorted(float(x) for x in samples_ms)
    p95_idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    avg = mean(ordered)
    std = pstdev(ordered) if len(ordered) > 1 else 0.0
    cv = (std / avg) if avg > 0 else float("inf")
    return SummaryStats(
        runtime_ms_mean=avg,
        runtime_ms_median=median(ordered),
        runtime_ms_p95=ordered[p95_idx],
        runtime_ms_stddev=std,
        runtime_ms_cv=cv,
    )


def compute_derived_metrics(bytes_moved: int, flops: float, runtime_ms_mean: float) -> dict[str, float]:
    runtime_seconds = runtime_ms_mean / 1000.0
    if runtime_seconds <= 0.0:
        raise ValueError("runtime_ms_mean must be positive.")

    effective_bandwidth_gbps = (bytes_moved / 1e9) / runtime_seconds
    effective_gflops = (flops / 1e9) / runtime_seconds
    arithmetic_intensity = (flops / bytes_moved) if bytes_moved > 0 else 0.0
    return {
        "effective_bandwidth_GBps": effective_bandwidth_gbps,
        "effective_GFLOPs": effective_gflops,
        "arithmetic_intensity": arithmetic_intensity,
    }


def default_unavailable_profiler_metrics() -> dict[str, str]:
    return {
        "occupancy": STATUS_UNAVAILABLE,
        "sm_utilization": STATUS_UNAVAILABLE,
        "l2_cache_hit_rate": STATUS_UNAVAILABLE,
    }
