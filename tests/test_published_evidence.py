"""Keep the published performance claims connected to intact raw evidence in CI."""
from pathlib import Path

from gpu_kernel_analyzer.artifacts import validate_run_directory
from gpu_kernel_analyzer.comparison import load_measured_run

ROOT = Path(__file__).resolve().parents[1] / "results"


def test_published_captures_are_internally_consistent():
    historical = load_measured_run(ROOT / "a100-2026-04-28")
    assert len(historical.rows) == 12
    h100 = ROOT / "h100-2026-09-22"
    for suite, count in [("baseline", 24), ("trial-1", 74), ("trial-2", 74), ("trial-3", 74), ("edges", 18)]:
        report = validate_run_directory(h100 / suite)
        assert report.ok, report.errors
        assert len(load_measured_run(h100 / suite).rows) == count
    manifests = [load_measured_run(h100 / f"trial-{i}").manifest for i in (1, 2, 3)]
    assert len({m["benchmark_binary_sha256"] for m in manifests}) == 1
    assert len({m["scenario_file_sha256"] for m in manifests}) == 1
