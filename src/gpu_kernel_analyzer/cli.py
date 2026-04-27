from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from .analysis import classify_bottleneck
from .artifacts import (
    PROVENANCE_COLUMNS,
    SUMMARY_COLUMNS,
    TIMING_COLUMNS,
    sha256_file,
    utc_now_iso,
    validate_run_directory,
    write_csv,
    write_json,
)
from .metrics import (
    STATUS_DERIVED,
    STATUS_MEASURED,
    STATUS_UNAVAILABLE,
    compute_derived_metrics,
    default_unavailable_profiler_metrics,
    summarize_runtime,
)
from .runner import run_binary_for_scenario
from .scenarios import load_and_expand_scenarios
from .plotting import generate_basic_plots
from .report import write_markdown_report
from .ncu import import_ncu_metrics
from .system_info import detect_ncu, runtime_environment


def _get_git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except Exception:
        return None


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}"


def _ensure_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(f"Binary output missing or invalid float field '{field}': {value}") from exc


def _ensure_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"Binary output missing or invalid int field '{field}': {value}") from exc


def run_sweep(args: argparse.Namespace) -> int:
    binary = Path(args.binary)
    scenario_file = Path(args.scenarios)
    run_dir = Path(args.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_and_expand_scenarios(scenario_file)
    if args.dry_run:
        print(f"Scenarios expanded: {len(scenarios)}")
        for scenario in scenarios:
            print(
                f"- kernel={scenario.kernel} size={scenario.problem_size} block={scenario.block_size} "
                f"warmups={scenario.warmups} repeats={scenario.repeats} verify={scenario.verify}"
            )
        return 0

    run_id = args.run_id or _new_run_id()
    timing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    first_device: dict[str, Any] | None = None

    for scenario in scenarios:
        binary_result = run_binary_for_scenario(binary=binary, scenario=scenario, interpreter=args.binary_interpreter)
        raw = binary_result.raw

        samples = binary_result.runtime_ms_samples
        stats = summarize_runtime(samples)
        bytes_moved = _ensure_int(raw.get("bytes_moved"), field="bytes_moved")
        flops = _ensure_float(raw.get("flops"), field="flops")
        derived = compute_derived_metrics(bytes_moved=bytes_moved, flops=flops, runtime_ms_mean=stats.runtime_ms_mean)

        for idx, sample_ms in enumerate(samples):
            timing_rows.append(
                {
                    "run_id": run_id,
                    "kernel": scenario.kernel,
                    "problem_size": scenario.problem_size,
                    "block_size": scenario.block_size,
                    "sample_index": idx,
                    "runtime_ms": sample_ms,
                }
            )

        summary_rows.append(
            {
                "run_id": run_id,
                "kernel": scenario.kernel,
                "problem_size": scenario.problem_size,
                "block_size": scenario.block_size,
                "warmups": scenario.warmups,
                "repeats": scenario.repeats,
                "verification_passed": bool(raw.get("verification_passed", True)),
                "runtime_ms_mean": stats.runtime_ms_mean,
                "runtime_ms_median": stats.runtime_ms_median,
                "runtime_ms_p95": stats.runtime_ms_p95,
                "runtime_ms_stddev": stats.runtime_ms_stddev,
                "runtime_ms_cv": stats.runtime_ms_cv,
                "effective_bandwidth_GBps": derived["effective_bandwidth_GBps"],
                "effective_GFLOPs": derived["effective_GFLOPs"],
                "arithmetic_intensity": derived["arithmetic_intensity"],
            }
        )

        metric_entries = [
            ("runtime_ms", stats.runtime_ms_mean, STATUS_MEASURED, "cuda_events"),
            (
                "effective_bandwidth_GBps",
                derived["effective_bandwidth_GBps"],
                STATUS_DERIVED,
                "bytes_moved/runtime_ms",
            ),
            ("effective_GFLOPs", derived["effective_GFLOPs"], STATUS_DERIVED, "flops/runtime_ms"),
            ("arithmetic_intensity", derived["arithmetic_intensity"], STATUS_DERIVED, "flops/bytes_moved"),
        ]
        for name, value, status, source in metric_entries:
            provenance_rows.append(
                {
                    "run_id": run_id,
                    "kernel": scenario.kernel,
                    "problem_size": scenario.problem_size,
                    "block_size": scenario.block_size,
                    "metric_name": name,
                    "metric_value": value,
                    "status": status,
                    "source": source,
                }
            )

        device = raw.get("device", {})
        if not isinstance(device, dict):
            device = {}
        if first_device is None:
            first_device = device

        metadata_available = bool(device.get("metadata_available", False))
        provenance_rows.append(
            {
                "run_id": run_id,
                "kernel": scenario.kernel,
                "problem_size": scenario.problem_size,
                "block_size": scenario.block_size,
                "metric_name": "device_metadata",
                "metric_value": json.dumps(device, sort_keys=True),
                "status": STATUS_MEASURED if metadata_available else STATUS_UNAVAILABLE,
                "source": "cuda_runtime_api",
            }
        )

        for metric_name, status in default_unavailable_profiler_metrics().items():
            provenance_rows.append(
                {
                    "run_id": run_id,
                    "kernel": scenario.kernel,
                    "problem_size": scenario.problem_size,
                    "block_size": scenario.block_size,
                    "metric_name": metric_name,
                    "metric_value": "",
                    "status": status,
                    "source": "nsight_compute_not_run",
                }
            )

    write_csv(run_dir / "timing_samples.csv", rows=timing_rows, fieldnames=TIMING_COLUMNS)
    write_csv(run_dir / "benchmark_summary.csv", rows=summary_rows, fieldnames=SUMMARY_COLUMNS)
    write_csv(run_dir / "metrics_provenance.csv", rows=provenance_rows, fieldnames=PROVENANCE_COLUMNS)

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "metrics_policy_version": "mvp_v1",
        "benchmark_binary": str(binary.resolve()),
        "benchmark_binary_sha256": sha256_file(binary),
        "scenario_file": str(scenario_file.resolve()),
        "scenario_file_sha256": sha256_file(scenario_file),
        "scenario_count": len(scenarios),
        "git_commit": _get_git_commit(),
        "runtime_environment": runtime_environment(),
        "device": first_device or {"metadata_available": False},
        "nsight_compute": detect_ncu(),
    }
    write_json(run_dir / "run_manifest.json", payload=manifest)

    print(f"Wrote run artifacts to: {run_dir.resolve()}")
    print(f"Scenarios executed: {len(scenarios)}")
    return 0


def validate_artifacts(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    report = validate_run_directory(run_dir)
    if report.ok:
        print(f"Validation passed for {run_dir.resolve()}")
        return 0
    print(f"Validation failed for {run_dir.resolve()}")
    for err in report.errors:
        print(f"- {err}")
    return 1


def analyze_quick(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("No rows in benchmark_summary.csv")
        return 0

    output_rows: list[dict[str, str]] = []
    for row in rows:
        result = classify_bottleneck(
            kernel=row["kernel"],
            arithmetic_intensity=float(row["arithmetic_intensity"]),
            effective_bandwidth_GBps=float(row["effective_bandwidth_GBps"]),
            effective_GFLOPs=float(row["effective_GFLOPs"]),
        )
        output_rows.append(
            {
                "kernel": row["kernel"],
                "problem_size": row["problem_size"],
                "block_size": row["block_size"],
                "likely_bottleneck": result.likely_bottleneck,
                "explanation": result.explanation,
            }
        )
        print(f"{row['kernel']} size={row['problem_size']} block={row['block_size']}: {result.explanation}")

    write_csv(
        run_dir / "analysis_heuristics.csv",
        rows=output_rows,
        fieldnames=["kernel", "problem_size", "block_size", "likely_bottleneck", "explanation"],
    )
    return 0


def analyze_full(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    # Reuse quick analysis output.
    quick_args = argparse.Namespace(run_dir=str(run_dir))
    analyze_quick(quick_args)
    plot_paths = generate_basic_plots(run_dir)
    report_path = write_markdown_report(run_dir)
    print(f"Generated report: {report_path}")
    for path in plot_paths:
        print(f"Generated plot: {path}")
    return 0


def profile_ncu_detect(_: argparse.Namespace) -> int:
    info = detect_ncu()
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def profile_ncu_import(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    ncu_csv = Path(args.ncu_csv)
    imported = import_ncu_metrics(
        run_dir=run_dir,
        ncu_csv=ncu_csv,
        source_tool=args.source_tool,
        source_file=args.source_file,
        metric_set=args.metric_set,
    )
    print(f"Imported {imported} Nsight Compute metric rows into {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU Kernel Performance Analyzer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    benchmark = sub.add_parser("benchmark", help="Run benchmark workflows.")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    sweep = benchmark_sub.add_parser("sweep", help="Run expanded benchmark sweeps and write artifacts.")
    sweep.add_argument("--binary", required=True, help="Path to benchmark binary executable.")
    sweep.add_argument("--binary-interpreter", default=None, help="Optional interpreter for script-based benchmark binaries.")
    sweep.add_argument("--scenarios", required=True, help="Path to benchmark scenarios YAML or JSON.")
    sweep.add_argument("--outdir", required=True, help="Output run directory.")
    sweep.add_argument("--run-id", default=None, help="Optional run id override.")
    sweep.add_argument("--dry-run", action="store_true", help="Expand scenarios without executing benchmark binary.")
    sweep.set_defaults(func=run_sweep)

    artifacts = sub.add_parser("artifacts", help="Validate or inspect run artifacts.")
    artifacts_sub = artifacts.add_subparsers(dest="artifacts_command", required=True)
    validate = artifacts_sub.add_parser("validate", help="Validate run artifact schema and metric policy.")
    validate.add_argument("--run-dir", required=True, help="Run directory with generated artifacts.")
    validate.set_defaults(func=validate_artifacts)

    analyze = sub.add_parser("analyze", help="Run lightweight analysis heuristics.")
    analyze_sub = analyze.add_subparsers(dest="analyze_command", required=True)
    quick = analyze_sub.add_parser("quick", help="Generate heuristic bottleneck summary from benchmark summary CSV.")
    quick.add_argument("--run-dir", required=True, help="Run directory with generated artifacts.")
    quick.set_defaults(func=analyze_quick)
    full = analyze_sub.add_parser("full", help="Generate heuristics, plots, and markdown report.")
    full.add_argument("--run-dir", required=True, help="Run directory with generated artifacts.")
    full.set_defaults(func=analyze_full)

    profile = sub.add_parser("profile", help="Optional profiler integration workflows.")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)
    ncu_detect = profile_sub.add_parser("ncu-detect", help="Detect Nsight Compute availability.")
    ncu_detect.set_defaults(func=profile_ncu_detect)
    ncu_import = profile_sub.add_parser(
        "ncu-import",
        help="Import normalized Nsight Compute CSV metrics into run provenance.",
    )
    ncu_import.add_argument("--run-dir", required=True, help="Run directory with generated artifacts.")
    ncu_import.add_argument(
        "--ncu-csv",
        required=True,
        help="CSV path with columns: kernel,problem_size,block_size,metric_name,metric_value",
    )
    ncu_import.add_argument(
        "--source-tool",
        required=True,
        choices=["ncu", "nsight_compute"],
        help="Profiler source tool identifier.",
    )
    ncu_import.add_argument(
        "--source-file",
        required=True,
        help="Path or identifier for the original Nsight source artifact.",
    )
    ncu_import.add_argument(
        "--metric-set",
        required=True,
        help="Profiler metric set identifier.",
    )
    ncu_import.set_defaults(func=profile_ncu_import)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
