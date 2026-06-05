from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .metrics import VALID_STATUSES
from .schemas import PROFILER_METRICS


TIMING_COLUMNS = [
    "run_id",
    "kernel",
    "problem_size",
    "block_size",
    "sample_index",
    "runtime_ms",
]

SUMMARY_COLUMNS = [
    "run_id",
    "kernel",
    "problem_size",
    "block_size",
    "warmups",
    "repeats",
    "verification_passed",
    "runtime_ms_mean",
    "runtime_ms_median",
    "runtime_ms_min",
    "runtime_ms_max",
    "runtime_ms_p95",
    "runtime_ms_stddev",
    "runtime_ms_cv",
    "effective_bandwidth_GBps",
    "effective_GFLOPs",
    "arithmetic_intensity",
]

PROVENANCE_COLUMNS = [
    "run_id",
    "kernel",
    "problem_size",
    "block_size",
    "metric_name",
    "metric_value",
    "status",
    "source",
]

DEFAULT_REQUIRED_FILES = [
    "run_manifest.json",
    "timing_samples.csv",
    "benchmark_summary.csv",
    "metrics_provenance.csv",
]


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _validate_manifest(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Invalid JSON in run_manifest.json: {exc}")
        return None
    if not isinstance(manifest, dict):
        errors.append("run_manifest.json must be a JSON object.")
        return None
    required = ["run_id", "created_at_utc", "benchmark_binary", "scenario_file", "metrics_policy_version"]
    for key in required:
        if key not in manifest:
            errors.append(f"run_manifest.json missing required key: {key}")
    return manifest


def _validate_columns(path: Path, required_columns: list[str], errors: list[str]) -> None:
    rows = read_csv_rows(path)
    if rows:
        present = set(rows[0].keys())
    else:
        with path.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        present = set(header) if header else set()
    missing = [col for col in required_columns if col not in present]
    if missing:
        errors.append(f"{path.name} missing required columns: {missing}")


def _validate_provenance_status(path: Path, errors: list[str], allow_profiler_measured: bool) -> None:
    rows = read_csv_rows(path)
    for idx, row in enumerate(rows, start=2):
        status = row.get("status", "")
        if status not in VALID_STATUSES:
            errors.append(f"{path.name} line {idx} has invalid status '{status}'")
    for idx, row in enumerate(rows, start=2):
        metric = row.get("metric_name", "")
        status = row.get("status", "")
        value = row.get("metric_value", "")
        if metric in PROFILER_METRICS:
            if status == "measured" and not allow_profiler_measured:
                errors.append(
                    f"{path.name} line {idx} metric '{metric}' cannot be measured unless Nsight data is parsed."
                )
            if status == "measured" and value == "":
                errors.append(f"{path.name} line {idx} metric '{metric}' is measured but has empty value.")
            if status == "measured":
                source = row.get("source", "")
                for token in ["source_tool=", "source_file=", "metric_set=", "import_timestamp="]:
                    if token not in source:
                        errors.append(
                            f"{path.name} line {idx} measured profiler metric missing traceable token '{token}'."
                        )


def _validate_nsight_manifest(nsight: dict[str, Any], errors: list[str]) -> None:
    required_keys = ["source_tool", "source_file", "metric_set", "import_timestamp", "source_csv"]
    for key in required_keys:
        value = nsight.get(key, "")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"run_manifest.json missing required nsight provenance field: {key}")
    source_tool = nsight.get("source_tool", "")
    if source_tool not in {"ncu", "nsight_compute"}:
        errors.append("run_manifest.json has invalid nsight source_tool; expected 'ncu' or 'nsight_compute'.")


def validate_run_directory(run_dir: Path) -> ValidationReport:
    errors: list[str] = []
    for filename in DEFAULT_REQUIRED_FILES:
        if not (run_dir / filename).exists():
            errors.append(f"Missing required file: {filename}")

    if errors:
        return ValidationReport(ok=False, errors=errors)

    manifest = _validate_manifest(run_dir / "run_manifest.json", errors)
    _validate_columns(run_dir / "timing_samples.csv", TIMING_COLUMNS, errors)
    _validate_columns(run_dir / "benchmark_summary.csv", SUMMARY_COLUMNS, errors)
    _validate_columns(run_dir / "metrics_provenance.csv", PROVENANCE_COLUMNS, errors)
    allow_profiler_measured = False
    if isinstance(manifest, dict):
        nsight = manifest.get("nsight_compute", {})
        if isinstance(nsight, dict):
            allow_profiler_measured = bool(nsight.get("enabled", False)) and str(nsight.get("status")) == "parsed_metrics"
            if allow_profiler_measured:
                _validate_nsight_manifest(nsight, errors)
    _validate_provenance_status(
        run_dir / "metrics_provenance.csv",
        errors,
        allow_profiler_measured=allow_profiler_measured,
    )
    return ValidationReport(ok=(len(errors) == 0), errors=errors)
