from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .advisor import build_recommendations, maybe_llm_summary, write_advisor_report
from .analysis import classify_bottleneck, compute_speedups
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
from .ncu import (
    build_ncu_raw_csv_command,
    get_metric_queries_for_set,
    import_ncu_metrics,
    normalize_ncu_raw_csv,
)
from .perf_model import (
    MIN_ROWS_FOR_CV,
    PREDICTION_COLUMNS,
    PerfModel,
    cross_validate,
    load_training_rows,
    prediction_rows,
    recommend_block_size,
    train_perf_model,
)
from .plotting import generate_basic_plots, generate_roofline_plot
from .report import write_markdown_report
from .runner import run_binary_for_scenario
from .scenarios import load_and_expand_scenarios, scenario_warnings
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
    for warning in scenario_warnings(scenarios):
        print(f"WARNING: {warning}")
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
                "runtime_ms_min": stats.runtime_ms_min,
                "runtime_ms_max": stats.runtime_ms_max,
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


SPEEDUP_COLUMNS = [
    "optimized_kernel",
    "baseline_kernel",
    "problem_size",
    "block_size",
    "baseline_runtime_ms_mean",
    "optimized_runtime_ms_mean",
    "speedup_runtime",
    "baseline_GFLOPs",
    "optimized_GFLOPs",
    "gflops_ratio",
]


def _write_speedup_csv(run_dir: Path) -> None:
    """Write analysis_speedup.csv pairing optimized kernels vs their baselines."""
    summary_path = run_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    speedups = compute_speedups(rows)
    out_rows = [
        {
            "optimized_kernel": s.optimized_kernel,
            "baseline_kernel": s.baseline_kernel,
            "problem_size": s.problem_size,
            "block_size": s.block_size,
            "baseline_runtime_ms_mean": s.baseline_runtime_ms_mean,
            "optimized_runtime_ms_mean": s.optimized_runtime_ms_mean,
            "speedup_runtime": s.speedup_runtime,
            "baseline_GFLOPs": s.baseline_GFLOPs,
            "optimized_GFLOPs": s.optimized_GFLOPs,
            "gflops_ratio": s.gflops_ratio,
        }
        for s in speedups
    ]
    write_csv(run_dir / "analysis_speedup.csv", rows=out_rows, fieldnames=SPEEDUP_COLUMNS)
    for s in speedups:
        print(
            f"speedup {s.optimized_kernel} vs {s.baseline_kernel} "
            f"size={s.problem_size}: {s.speedup_runtime:.2f}x faster (runtime mean)"
        )


def analyze_full(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    # Reuse quick analysis output.
    quick_args = argparse.Namespace(run_dir=str(run_dir))
    analyze_quick(quick_args)
    _write_speedup_csv(run_dir)
    plot_paths = generate_basic_plots(run_dir)
    roofline_path = generate_roofline_plot(
        run_dir,
        peak_gflops=getattr(args, "peak_gflops", None),
        peak_bandwidth_GBps=getattr(args, "peak_bandwidth_gbps", None),
    )
    if roofline_path is not None:
        plot_paths.append(roofline_path)
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


def profile_ncu_normalize(args: argparse.Namespace) -> int:
    normalized_rows = normalize_ncu_raw_csv(
        raw_csv=Path(args.raw_csv),
        normalized_csv=Path(args.out_csv),
        kernel=args.kernel,
        problem_size=int(args.problem_size),
        block_size=int(args.block_size),
        metric_set=args.metric_set,
        metric_config_path=Path(args.metric_config),
    )
    metrics = ",".join(row["metric_name"] for row in normalized_rows)
    print(f"Wrote {len(normalized_rows)} normalized Nsight rows to {Path(args.out_csv).resolve()}")
    print(f"Extracted metrics: {metrics}")
    return 0


def profile_ncu_plan(args: argparse.Namespace) -> int:
    metric_config = Path(args.metric_config)
    metric_queries = get_metric_queries_for_set(args.metric_set, metric_config)
    raw_cmd = build_ncu_raw_csv_command(
        binary=Path(args.binary),
        kernel=args.kernel,
        problem_size=int(args.problem_size),
        block_size=int(args.block_size),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        verify=bool(args.verify),
        metric_queries=metric_queries,
        raw_csv_out=Path(args.raw_csv_out),
    )
    source_file = args.source_file if args.source_file else str(Path(args.raw_csv_out))
    import_cmd = " ".join(
        [
            "python -m gpu_kernel_analyzer profile ncu-import",
            f"--run-dir {shlex.quote(args.run_dir)}",
            "--source-tool ncu",
            f"--source-file {shlex.quote(source_file)}",
            f"--metric-set {shlex.quote(args.metric_set)}",
            f"--ncu-csv {shlex.quote(args.normalized_csv)}",
        ]
    )
    print("NCU raw capture command:")
    print(raw_cmd)
    print()
    print("Normalized import command:")
    print(import_cmd)
    print()
    print("Normalization contract:")
    print("kernel,problem_size,block_size,metric_name,metric_value")
    print(
        "Allowed metric_name values: occupancy,sm_utilization,memory_throughput_pct,"
        "l2_throughput_pct,l2_cache_hit_rate"
    )
    return 0


_GPU_DATA_HINT = (
    "To add real measured data, run a sweep on a GPU machine, e.g.:\n"
    "  python -m gpu_kernel_analyzer benchmark sweep --binary build/gpu_benchmark "
    "--scenarios configs/benchmark_scenarios.yaml --outdir outputs/run_gpu"
)


def model_train(args: argparse.Namespace) -> int:
    run_dirs = [Path(d) for d in args.run_dir]
    rows = load_training_rows(run_dirs)
    if not rows:
        print("No usable benchmark_summary rows found in the given run directories.")
        print(_GPU_DATA_HINT)
        return 1

    model = train_perf_model(rows)
    model_out = Path(args.model_out)
    write_json(model_out, model.to_dict())
    print(f"Trained ridge_log_linear model (backend={model.backend}) on {model.training_rows} rows.")
    print(f"Saved model to: {model_out.resolve()}")

    cv = cross_validate(rows)
    if not cv["enough_data"]:
        print(
            f"WARNING: only {cv['n_rows']} training rows (< {MIN_ROWS_FOR_CV}). "
            "Cross-validation metrics below are not statistically meaningful; treat the model as a demonstration."
        )
        print(_GPU_DATA_HINT)
    for target, metrics in cv["targets"].items():
        r2 = metrics["r2"]
        r2_text = "nan" if r2 != r2 else f"{r2:.3f}"
        print(f"CV[{target}]: MAE={metrics['mae']:.5f} R2={r2_text} (n={metrics['n']}, leave-one-out)")
    return 0


def _write_predictions(predictions_out: str | None, rows: list[dict[str, Any]]) -> None:
    if not predictions_out:
        return
    write_csv(Path(predictions_out), rows=rows, fieldnames=PREDICTION_COLUMNS)
    print(f"Wrote {len(rows)} predicted rows to: {Path(predictions_out).resolve()}")


def model_predict(args: argparse.Namespace) -> int:
    model = PerfModel.from_dict(json.loads(Path(args.model).read_text(encoding="utf-8")))
    predicted = model.predict(args.kernel, args.problem_size, args.block_size)
    source = f"perf_model:{model.backend};model={Path(args.model).name}"
    payload = {
        "kernel": args.kernel,
        "problem_size": args.problem_size,
        "block_size": args.block_size,
        "status": "predicted",
        "predicted": predicted,
        "model_source": source,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    _write_predictions(
        args.predictions_out,
        prediction_rows(args.kernel, args.problem_size, args.block_size, predicted, source),
    )
    return 0


def model_recommend_block_size(args: argparse.Namespace) -> int:
    model = PerfModel.from_dict(json.loads(Path(args.model).read_text(encoding="utf-8")))
    candidates = [int(c.strip()) for c in args.candidates.split(",") if c.strip()]
    rec = recommend_block_size(model, args.kernel, args.problem_size, candidates)
    print(
        f"Recommended block_size for {rec.kernel} size={rec.problem_size}: {rec.recommended_block_size} "
        f"(predicted runtime {rec.predicted_runtime_ms:.5f} ms)"
    )
    for block_size, runtime in sorted(rec.candidates, key=lambda item: item[1]):
        print(f"  block={block_size}: predicted runtime {runtime:.5f} ms")
    if not rec.data_backed:
        print(
            f"WARNING: training data did not vary block_size for '{rec.kernel}', so this recommendation is "
            "an extrapolation rather than a data-backed choice. Add runs that sweep block sizes."
        )
        print(_GPU_DATA_HINT)
    source = f"perf_model:{model.backend};model={Path(args.model).name}"
    pred_rows: list[dict[str, Any]] = []
    for block_size, _ in rec.candidates:
        predicted = model.predict(rec.kernel, rec.problem_size, block_size)
        pred_rows.extend(prediction_rows(rec.kernel, rec.problem_size, block_size, predicted, source))
    _write_predictions(args.predictions_out, pred_rows)
    return 0


def advise(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    recommendations = build_recommendations(run_dir)
    if getattr(args, "llm", False):
        summary = maybe_llm_summary(recommendations)
        if summary is None:
            print("LLM summary requested but no LLM backend is configured. Using rule-based recommendations.")
        else:  # pragma: no cover - no bundled backend
            print(summary)
            print()
    for rec in recommendations:
        print(rec)
    report_path = write_advisor_report(run_dir, recommendations)
    print(f"Wrote advisor report: {report_path.resolve()}")
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
    full.add_argument(
        "--peak-gflops",
        type=float,
        default=None,
        help="Optional real peak FP32 GFLOPs for this GPU; draws a compute roof on the roofline plot.",
    )
    full.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        default=None,
        help="Optional real peak DRAM bandwidth (GB/s) for this GPU; draws a memory roof on the roofline plot.",
    )
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
    ncu_normalize = profile_sub.add_parser(
        "ncu-normalize",
        help="Normalize raw Nsight CSV into import format for one scenario.",
    )
    ncu_normalize.add_argument("--raw-csv", required=True, help="Raw Nsight CSV path from ncu --page raw.")
    ncu_normalize.add_argument(
        "--out-csv",
        required=True,
        help="Output normalized CSV path with kernel,problem_size,block_size,metric_name,metric_value.",
    )
    ncu_normalize.add_argument("--kernel", required=True, help="Scenario kernel identifier.")
    ncu_normalize.add_argument("--problem-size", required=True, type=int, help="Scenario problem size.")
    ncu_normalize.add_argument("--block-size", required=True, type=int, help="Scenario block size.")
    ncu_normalize.add_argument(
        "--metric-set",
        default="default_profiler_set",
        help="Metric-set name from configs/ncu_metric_sets.yaml.",
    )
    ncu_normalize.add_argument(
        "--metric-config",
        default="configs/ncu_metric_sets.yaml",
        help="Path to ncu metric-set config YAML.",
    )
    ncu_normalize.set_defaults(func=profile_ncu_normalize)
    ncu_plan = profile_sub.add_parser(
        "ncu-plan",
        help="Print exact Nsight raw capture and import commands for one benchmark scenario.",
    )
    ncu_plan.add_argument("--binary", required=True, help="Path to benchmark binary on CUDA machine.")
    ncu_plan.add_argument("--kernel", required=True, help="Kernel name (e.g. vector_add, gemm_tiled).")
    ncu_plan.add_argument("--problem-size", required=True, type=int, help="Scenario problem size.")
    ncu_plan.add_argument("--block-size", required=True, type=int, help="Scenario block size.")
    ncu_plan.add_argument("--warmups", required=True, type=int, help="Warmup count for selected scenario.")
    ncu_plan.add_argument("--repeats", required=True, type=int, help="Repeat count for selected scenario.")
    ncu_plan.add_argument("--verify", action="store_true", help="Append --verify to benchmark command.")
    ncu_plan.add_argument(
        "--metric-set",
        default="default_profiler_set",
        help="Metric-set name from configs/ncu_metric_sets.yaml.",
    )
    ncu_plan.add_argument(
        "--metric-config",
        default="configs/ncu_metric_sets.yaml",
        help="Path to ncu metric-set config YAML.",
    )
    ncu_plan.add_argument(
        "--raw-csv-out",
        required=True,
        help="Output path for Nsight raw CSV capture.",
    )
    ncu_plan.add_argument(
        "--normalized-csv",
        required=True,
        help="Path to normalized CSV (kernel,problem_size,block_size,metric_name,metric_value).",
    )
    ncu_plan.add_argument(
        "--run-dir",
        required=True,
        help="Existing run directory where imported metrics will be attached.",
    )
    ncu_plan.add_argument(
        "--source-file",
        default=None,
        help="Optional source-file metadata override for ncu-import (defaults to --raw-csv-out).",
    )
    ncu_plan.set_defaults(func=profile_ncu_plan)

    model = sub.add_parser("model", help="Train and use the performance prediction model.")
    model_sub = model.add_subparsers(dest="model_command", required=True)

    model_train_parser = model_sub.add_parser(
        "train",
        help="Train a performance model from one or more run directories.",
    )
    model_train_parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing benchmark_summary.csv (repeatable to pool runs).",
    )
    model_train_parser.add_argument("--model-out", required=True, help="Output path for the trained model JSON.")
    model_train_parser.set_defaults(func=model_train)

    model_predict_parser = model_sub.add_parser(
        "predict",
        help="Predict runtime and bandwidth for a kernel configuration.",
    )
    model_predict_parser.add_argument("--model", required=True, help="Path to a trained model JSON.")
    model_predict_parser.add_argument("--kernel", required=True, help="Kernel name.")
    model_predict_parser.add_argument("--problem-size", required=True, type=int, help="Problem size.")
    model_predict_parser.add_argument("--block-size", required=True, type=int, help="Block size.")
    model_predict_parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional path to write predicted rows (model_predictions.csv).",
    )
    model_predict_parser.set_defaults(func=model_predict)

    model_recommend_parser = model_sub.add_parser(
        "recommend-block-size",
        help="Rank candidate block sizes by predicted runtime.",
    )
    model_recommend_parser.add_argument("--model", required=True, help="Path to a trained model JSON.")
    model_recommend_parser.add_argument("--kernel", required=True, help="Kernel name.")
    model_recommend_parser.add_argument("--problem-size", required=True, type=int, help="Problem size.")
    model_recommend_parser.add_argument(
        "--candidates",
        required=True,
        help="Comma-separated candidate block sizes, e.g. 64,128,256,512.",
    )
    model_recommend_parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional path to write predicted rows (model_predictions.csv).",
    )
    model_recommend_parser.set_defaults(func=model_recommend_block_size)

    advise_parser = sub.add_parser("advise", help="Generate rule-based tuning recommendations for a run.")
    advise_parser.add_argument("--run-dir", required=True, help="Run directory with generated artifacts.")
    advise_parser.add_argument(
        "--llm",
        action="store_true",
        help="Optional: request an LLM summary if a backend is configured (no backend bundled; falls back to rules).",
    )
    advise_parser.set_defaults(func=advise)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
