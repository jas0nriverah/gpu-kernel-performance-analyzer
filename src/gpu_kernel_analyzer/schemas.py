from __future__ import annotations

from dataclasses import dataclass


METRIC_RUNTIME_MS = "runtime_ms"
METRIC_EFFECTIVE_BANDWIDTH = "effective_bandwidth_GBps"
METRIC_EFFECTIVE_GFLOPS = "effective_GFLOPs"
METRIC_ARITHMETIC_INTENSITY = "arithmetic_intensity"
METRIC_DEVICE_METADATA = "device_metadata"

PROFILER_METRICS = {
    "occupancy",
    "sm_utilization",
    "memory_throughput_pct",
    "l2_throughput_pct",
    "l2_cache_hit_rate",
}


@dataclass(frozen=True)
class MetricsPolicy:
    version: str
    default_measured: tuple[str, ...]
    default_derived_estimates: tuple[str, ...]
    default_unavailable_profiler_metrics: tuple[str, ...]


MVP_METRICS_POLICY = MetricsPolicy(
    version="mvp_v1",
    default_measured=(METRIC_RUNTIME_MS, METRIC_DEVICE_METADATA),
    default_derived_estimates=(
        METRIC_EFFECTIVE_BANDWIDTH,
        METRIC_EFFECTIVE_GFLOPS,
        METRIC_ARITHMETIC_INTENSITY,
    ),
    default_unavailable_profiler_metrics=tuple(sorted(PROFILER_METRICS)),
)
