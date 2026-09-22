"""Compare measured runs by scenario, with raw-sample checks and explicit CI policy."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .artifacts import write_csv, write_json

ScenarioKey = tuple[str, int, int]


@dataclass(frozen=True)
class Run:
    manifest: dict[str, Any]
    rows: dict[ScenarioKey, dict[str, str]]


def _key(row: dict[str, str]) -> ScenarioKey:
    return row["kernel"], int(row["problem_size"]), int(row["block_size"])


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_measured_run(path: Path) -> Run:
    """Accept current and historical CSVs; recompute claimed statistics from raw timings.

    Historical runs need not contain newer min/max columns. They must still have
    positive samples, matching run/scenario IDs, contiguous indices, and passed checks.
    """
    manifest = json.loads((path / "run_manifest.json").read_text())
    if not isinstance(manifest, dict) or not manifest.get("run_id"):
        raise ValueError(f"Invalid run manifest: {path}")
    rows: dict[ScenarioKey, dict[str, str]] = {}
    for row in _csv(path / "benchmark_summary.csv"):
        key = _key(row)
        if key in rows:
            raise ValueError(f"Duplicate summary scenario: {key}")
        if row["run_id"] != manifest["run_id"] or row["verification_passed"].lower() != "true":
            raise ValueError(f"Unverified or mismatched run: {key}")
        rows[key] = row
    if not rows:
        raise ValueError("Cannot compare an empty run")
    if manifest.get("scenario_count", len(rows)) != len(rows):
        raise ValueError("Manifest scenario count does not match summary")
    samples: dict[ScenarioKey, dict[int, float]] = {key: {} for key in rows}
    for row in _csv(path / "timing_samples.csv"):
        key = _key(row)
        if key not in rows or row["run_id"] != manifest["run_id"]:
            raise ValueError(f"Unexpected raw timing scenario or run: {key}")
        index, value = int(row["sample_index"]), float(row["runtime_ms"])
        if index in samples[key] or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Duplicate index or invalid timing: {key}")
        samples[key][index] = value
    for key, row in rows.items():
        repeats = int(row["repeats"])
        if repeats < 1 or set(samples[key]) != set(range(repeats)):
            raise ValueError(f"Raw sample count/indices do not match repeats: {key}")
        values = list(samples[key].values())
        avg = mean(values)
        for field, expected in {
            "runtime_ms_mean": avg,
            "runtime_ms_median": median(values),
            "runtime_ms_cv": pstdev(values) / avg,
        }.items():
            if not math.isclose(float(row[field]), expected, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f"Summary {field} disagrees with raw samples: {key}")
    return Run(manifest, rows)


def compare_runs(
    baseline: Run, candidate: Run, *, threshold_pct: float = 5.0,
    statistic: str = "median", max_cv: float = 0.10, allow_device_mismatch: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(threshold_pct) or threshold_pct < 0:
        raise ValueError("threshold_pct must be finite and nonnegative")
    if not math.isfinite(max_cv) or max_cv < 0:
        raise ValueError("max_cv must be finite and nonnegative")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be mean or median")
    device_fields = ("name", "compute_capability_major", "compute_capability_minor", "multiprocessor_count", "total_global_mem_bytes")
    devices = [run.manifest.get("device", {}) for run in (baseline, candidate)]
    if any(not d.get("metadata_available") or not d.get("name") for d in devices):
        raise ValueError("Device metadata is required for comparison")
    mismatch = any(devices[0].get(field) != devices[1].get(field) for field in device_fields)
    if mismatch and not allow_device_mismatch:
        raise ValueError("Different GPUs: pass --allow-device-mismatch for a descriptive comparison")
    if baseline.manifest.get("metrics_policy_version") != candidate.manifest.get("metrics_policy_version"):
        raise ValueError("Metric policy versions differ")
    matched = sorted(baseline.rows.keys() & candidate.rows.keys())
    if not matched:
        raise ValueError("No matching scenarios")
    warnings = []
    if mismatch:
        warnings.append("Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.")
    for field in ("git_commit", "benchmark_binary_sha256", "scenario_file_sha256", "runtime_environment", "source_sha256"):
        if baseline.manifest.get(field) != candidate.manifest.get(field):
            warnings.append(f"Run metadata differs: {field}.")
    results = []
    protocol_mismatches = []
    for key in matched:
        before, after = baseline.rows[key], candidate.rows[key]
        protocol_changed = any(before[field] != after[field] for field in ("warmups", "repeats"))
        if protocol_changed:
            protocol_mismatches.append(list(key))
        a, b = float(before[f"runtime_ms_{statistic}"]), float(after[f"runtime_ms_{statistic}"])
        change = (b / a - 1.0) * 100.0
        noisy = max(float(before["runtime_ms_cv"]), float(after["runtime_ms_cv"])) > max_cv
        results.append({
            "kernel": key[0], "problem_size": key[1], "block_size": key[2],
            "baseline_ms": a, "candidate_ms": b, "speedup": a / b,
            "change_pct": change,
            "status": "regression" if b > a * (1 + threshold_pct / 100) else "improvement" if b < a * (1 - threshold_pct / 100) else "within_threshold",
            "baseline_cv": float(before["runtime_ms_cv"]), "candidate_cv": float(after["runtime_ms_cv"]),
            "noisy": noisy, "protocol_changed": protocol_changed,
        })
    missing = [list(key) for key in sorted(baseline.rows.keys() - candidate.rows.keys())]
    added = [list(key) for key in sorted(candidate.rows.keys() - baseline.rows.keys())]
    if protocol_mismatches:
        warnings.append("Warmup/repeat counts differ for matched scenarios; CI gating is blocked.")
    gate_reasons = []
    if mismatch:
        gate_reasons.append("GPU metadata differs")
    if missing:
        gate_reasons.append(f"{len(missing)} baseline scenarios are missing")
    if protocol_mismatches:
        gate_reasons.append(f"{len(protocol_mismatches)} scenarios changed warmups/repeats")
    noisy_count = sum(row["noisy"] for row in results)
    regression_count = sum(row["status"] == "regression" for row in results)
    if noisy_count:
        gate_reasons.append(f"{noisy_count} matched scenarios exceed the CV limit ({max_cv:g})")
    if regression_count:
        gate_reasons.append(f"{regression_count} matched scenarios exceed the slowdown threshold")
    return {
        "baseline_run_id": baseline.manifest["run_id"], "candidate_run_id": candidate.manifest["run_id"],
        "baseline_device": devices[0]["name"], "candidate_device": devices[1]["name"],
        "statistic": statistic, "threshold_pct": threshold_pct, "max_cv": max_cv,
        "device_mismatch": mismatch, "warnings": warnings, "rows": results,
        "missing_scenarios": missing, "added_scenarios": added,
        "regressions": regression_count, "noisy_scenarios": noisy_count, "gate_reasons": gate_reasons,
        # Noise is not evidence of a pass. Require rerunning unstable measurements.
        "gate_passed": not gate_reasons,
    }


def write_comparison(result: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "comparison.json", result)
    write_csv(outdir / "comparison.csv", result["rows"], list(result["rows"][0]))
    lines = [
        "# Benchmark comparison", "",
        f"{result['baseline_device']} → {result['candidate_device']}", "",
        f"Statistic: **{result['statistic']}**. Threshold: **{result['threshold_pct']:g}%**. "
        "Speedup = baseline / candidate runtime; positive change means slower.", "",
        "These are descriptive timing comparisons, not statistical significance tests. "
        "CV flags within-run variability; it does not measure uncertainty across independent runs.", "",
        f"Matched: {len(result['rows'])}; missing: {len(result['missing_scenarios'])}; "
        f"added: {len(result['added_scenarios'])}; regressions: {result['regressions']}.", "",
        f"CI gate: **{'PASS' if result['gate_passed'] else 'FAIL / not eligible'}**.", "",
        *[f"- Gate: {reason}." for reason in result["gate_reasons"]],
        *[f"- {warning}" for warning in result["warnings"]], "",
        "| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['kernel']} | {row['problem_size']} | {row['block_size']} | "
            f"{row['baseline_ms']:.6f} | {row['candidate_ms']:.6f} | {row['speedup']:.2f}× | "
            f"{row['change_pct']:+.1f}% | {row['status']} | {'yes' if row['noisy'] else 'no'} |"
        )
    for label, key in (("Missing from candidate", "missing_scenarios"), ("Added in candidate", "added_scenarios")):
        if result[key]:
            lines.extend(["", f"## {label}", "", *[f"- `{k}`, size {n}, block {b}" for k, n, b in result[key]]])
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_command(args: Any) -> int:
    try:
        output = Path(args.outdir).resolve()
        inputs = [Path(args.baseline).resolve(), Path(args.candidate).resolve()]
        if any(output == p or output in p.parents or p in output.parents for p in inputs):
            raise ValueError("Comparison output must be separate from input run directories")
        result = compare_runs(
            *(load_measured_run(p) for p in inputs), threshold_pct=args.threshold_pct,
            statistic=args.statistic, max_cv=args.max_cv, allow_device_mismatch=args.allow_device_mismatch,
        )
        write_comparison(result, output)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Comparison failed: {exc}")
        return 2
    print(f"Compared {len(result['rows'])} scenarios; {result['regressions']} regressions. Report: {output / 'REPORT.md'}")
    return 1 if args.fail_on_regression and not result["gate_passed"] else 0
