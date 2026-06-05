from __future__ import annotations

import csv
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_csv, write_csv
from .schemas import PROFILER_METRICS

try:
    import yaml
except Exception:  # pragma: no cover - optional import guard
    yaml = None


ALLOWED_SOURCE_TOOLS = {"ncu", "nsight_compute"}


def _required_text(value: str, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"Missing required provenance field: {field}")
    return text


def _load_metric_set(metric_set: str, metric_config_path: Path) -> dict[str, Any]:
    metric_set = _required_text(metric_set, "metric_set")
    if yaml is None:
        raise RuntimeError("PyYAML is required for loading ncu metric-set config.")
    if not metric_config_path.exists():
        raise FileNotFoundError(f"ncu metric-set config not found: {metric_config_path}")

    payload = yaml.safe_load(metric_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ncu metric-set config must be a YAML mapping.")

    metric_sets = payload.get("metric_sets", {})
    if not isinstance(metric_sets, dict) or metric_set not in metric_sets:
        available = sorted(metric_sets.keys()) if isinstance(metric_sets, dict) else []
        raise ValueError(f"Unknown metric_set '{metric_set}'. Available sets: {available}")

    selected = metric_sets[metric_set]
    if not isinstance(selected, dict):
        raise ValueError(f"Metric set '{metric_set}' must be a mapping.")
    return selected


def get_metric_queries_for_set(metric_set: str, metric_config_path: Path) -> list[str]:
    selected = _load_metric_set(metric_set, metric_config_path)
    ncu_metrics = selected.get("ncu_metrics", [])
    queries: list[str] = []
    if isinstance(ncu_metrics, list):
        for metric in ncu_metrics:
            queries.append(_required_text(str(metric), "ncu_metrics"))
        return queries
    if isinstance(ncu_metrics, dict):
        for metric in ncu_metrics.values():
            queries.append(_required_text(str(metric), "ncu_metrics"))
        # Backward-compatibility for old config style.
        return queries
    raise ValueError(f"Metric set '{metric_set}' must define ncu_metrics as a list or mapping.")


def get_normalization_sources_for_set(metric_set: str, metric_config_path: Path) -> dict[str, list[str]]:
    selected = _load_metric_set(metric_set, metric_config_path)
    normalized_order = selected.get("normalized_metrics", [])
    if not isinstance(normalized_order, list) or not normalized_order:
        raise ValueError(f"Metric set '{metric_set}' must define non-empty normalized_metrics.")

    raw_sources = selected.get("normalized_metric_sources", {})
    if not isinstance(raw_sources, dict):
        raise ValueError(f"Metric set '{metric_set}' must define normalized_metric_sources mapping.")

    normalized_sources: dict[str, list[str]] = {}
    for metric_name in normalized_order:
        metric_name = _required_text(str(metric_name), "normalized_metrics")
        if metric_name not in PROFILER_METRICS:
            raise ValueError(f"Metric '{metric_name}' in metric set is not a supported profiler metric.")
        source_columns = raw_sources.get(metric_name, [])
        if isinstance(source_columns, str):
            source_columns = [source_columns]
        if not isinstance(source_columns, list):
            raise ValueError(f"Metric '{metric_name}' must map to a list of raw metric column names.")
        normalized_sources[metric_name] = [_required_text(str(col), f"{metric_name}_source") for col in source_columns]
    return normalized_sources


def _is_units_row(row: list[str]) -> bool:
    return len(row) >= 2 and row[0].strip() == "" and row[1].strip() == ""


def _parse_raw_ncu_records(raw_csv: Path) -> list[dict[str, str]]:
    if not raw_csv.exists():
        raise FileNotFoundError(f"NCU raw CSV not found: {raw_csv}")
    filtered_rows: list[list[str]] = []
    with raw_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if len(row) == 1 and row[0].startswith("==PROF=="):
                continue
            filtered_rows.append(row)
    if not filtered_rows:
        raise ValueError("NCU raw CSV had no parseable rows.")

    header_idx = next((i for i, row in enumerate(filtered_rows) if "Kernel Name" in row and "ID" in row), None)
    if header_idx is None:
        raise ValueError("NCU raw CSV is missing expected header row with 'Kernel Name'.")

    header = filtered_rows[header_idx]
    payload_rows = filtered_rows[header_idx + 1 :]
    if not payload_rows:
        raise ValueError("NCU raw CSV did not contain metric payload rows.")
    if payload_rows and _is_units_row(payload_rows[0]):
        payload_rows = payload_rows[1:]

    records: list[dict[str, str]] = []
    for row in payload_rows:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        records.append({header[i]: row[i] for i in range(len(header))})
    if not records:
        raise ValueError("NCU raw CSV payload was empty after removing metadata rows.")
    return records


def _parse_block_size_x(block_size_field: str) -> int | None:
    text = block_size_field.strip()
    if not text:
        return None
    tuple_match = re.match(r"\(\s*(\d+)", text)
    if tuple_match:
        return int(tuple_match.group(1))
    if text.isdigit():
        return int(text)
    return None


def _parse_numeric_metric(value: str) -> float | None:
    text = value.strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_ncu_raw_csv(
    *,
    raw_csv: Path,
    normalized_csv: Path,
    kernel: str,
    problem_size: int,
    block_size: int,
    metric_set: str,
    metric_config_path: Path,
) -> list[dict[str, str]]:
    kernel = _required_text(kernel, "kernel")
    metric_sources = get_normalization_sources_for_set(metric_set, metric_config_path)
    raw_records = _parse_raw_ncu_records(raw_csv)

    scenario_rows: list[dict[str, str]] = []
    for row in raw_records:
        kernel_name = row.get("Kernel Name", "")
        block_x = _parse_block_size_x(row.get("Block Size", ""))
        if block_x is None:
            block_x = _parse_block_size_x(row.get("launch__block_dim_x", ""))
        if block_x is None:
            continue
        if kernel.lower() in kernel_name.lower() and block_x == block_size:
            scenario_rows.append(row)

    if not scenario_rows:
        discovered = sorted({row.get("Kernel Name", "") for row in raw_records})
        raise RuntimeError(
            "No Nsight raw row matched selected scenario. "
            f"kernel={kernel}, block_size={block_size}, discovered_kernels={discovered}"
        )
    if len(scenario_rows) > 1:
        raise RuntimeError(
            "Ambiguous Nsight raw rows for selected scenario. "
            f"kernel={kernel}, block_size={block_size}, matches={len(scenario_rows)}"
        )
    selected = scenario_rows[0]

    normalized_rows: list[dict[str, str]] = []
    for logical_metric, candidate_columns in metric_sources.items():
        metric_value_text: str | None = None
        for candidate in candidate_columns:
            raw_value = selected.get(candidate, "")
            parsed = _parse_numeric_metric(raw_value)
            if parsed is not None:
                metric_value_text = raw_value.strip().replace("%", "")
                break
        if metric_value_text is None:
            continue
        normalized_rows.append(
            {
                "kernel": kernel,
                "problem_size": str(problem_size),
                "block_size": str(block_size),
                "metric_name": logical_metric,
                "metric_value": metric_value_text,
            }
        )

    if not normalized_rows:
        raise RuntimeError("No configured profiler metrics were found in the selected Nsight raw row.")

    write_csv(
        normalized_csv,
        rows=normalized_rows,
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )
    return normalized_rows


def build_ncu_raw_csv_command(
    *,
    binary: Path,
    kernel: str,
    problem_size: int,
    block_size: int,
    warmups: int,
    repeats: int,
    verify: bool,
    metric_queries: list[str],
    raw_csv_out: Path,
) -> str:
    metric_queries = [q.strip() for q in metric_queries if q.strip()]
    if not metric_queries:
        raise ValueError("metric_queries must contain at least one Nsight metric.")
    kernel = _required_text(kernel, "kernel")
    binary_text = _required_text(str(binary), "binary")
    raw_csv_text = _required_text(str(raw_csv_out), "raw_csv_out")

    cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metric_queries),
        binary_text,
        "--kernel",
        kernel,
        "--problem-size",
        str(problem_size),
        "--block-size",
        str(block_size),
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
    ]
    if verify:
        cmd.append("--verify")
    return " ".join(shlex.quote(piece) for piece in cmd) + f" > {shlex.quote(raw_csv_text)}"


def import_ncu_metrics(
    run_dir: Path,
    ncu_csv: Path,
    *,
    source_tool: str,
    source_file: str,
    metric_set: str,
) -> int:
    """
    Import profiler metrics from a normalized CSV with columns:
    kernel,problem_size,block_size,metric_name,metric_value
    """
    if not ncu_csv.exists():
        raise FileNotFoundError(f"NCU CSV not found: {ncu_csv}")

    source_tool = _required_text(source_tool, "source_tool")
    source_file = _required_text(source_file, "source_file")
    metric_set = _required_text(metric_set, "metric_set")
    if source_tool not in ALLOWED_SOURCE_TOOLS:
        raise ValueError(f"Unsupported source_tool '{source_tool}'. Allowed: {sorted(ALLOWED_SOURCE_TOOLS)}")

    ncu_rows = read_csv(ncu_csv)
    required = {"kernel", "problem_size", "block_size", "metric_name", "metric_value"}
    if ncu_rows and not required.issubset(ncu_rows[0].keys()):
        raise ValueError(
            "NCU CSV must contain columns: kernel,problem_size,block_size,metric_name,metric_value"
        )

    provenance_path = run_dir / "metrics_provenance.csv"
    prov_rows = read_csv(provenance_path)
    summary_rows = read_csv(run_dir / "benchmark_summary.csv")
    imported = 0
    imported_at = datetime.now(timezone.utc).isoformat()
    for ncu_row in ncu_rows:
        kernel = _required_text(ncu_row.get("kernel", ""), "kernel")
        problem_size = _required_text(ncu_row.get("problem_size", ""), "problem_size")
        block_size = _required_text(ncu_row.get("block_size", ""), "block_size")
        metric_name = _required_text(ncu_row.get("metric_name", ""), "metric_name")
        metric_value = _required_text(ncu_row.get("metric_value", ""), "metric_value")

        if metric_name not in PROFILER_METRICS:
            raise ValueError(f"Unsupported profiler metric '{metric_name}' in NCU import.")

        scenario_matches = [
            row
            for row in summary_rows
            if row.get("kernel") == kernel
            and row.get("problem_size") == problem_size
            and row.get("block_size") == block_size
        ]
        if not scenario_matches:
            raise RuntimeError(
                "No matching benchmark scenario for Nsight metric row: "
                f"kernel={kernel}, problem_size={problem_size}, block_size={block_size}, metric={metric_name}"
            )
        if len(scenario_matches) > 1:
            raise RuntimeError(
                "Ambiguous Nsight metric mapping; multiple matching benchmark scenarios found for: "
                f"kernel={kernel}, problem_size={problem_size}, block_size={block_size}, metric={metric_name}"
            )

        matches = [
            row
            for row in prov_rows
            if row.get("kernel") == kernel
            and row.get("problem_size") == problem_size
            and row.get("block_size") == block_size
            and row.get("metric_name") == metric_name
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "Ambiguous Nsight metric mapping; multiple matching benchmark scenarios found for: "
                f"kernel={kernel}, problem_size={problem_size}, block_size={block_size}, metric={metric_name}"
            )
        if not matches:
            scenario_row = scenario_matches[0]
            seeded = {
                "run_id": scenario_row.get("run_id", ""),
                "kernel": kernel,
                "problem_size": problem_size,
                "block_size": block_size,
                "metric_name": metric_name,
                "metric_value": "",
                "status": "unavailable",
                "source": "nsight_compute_not_run",
            }
            prov_rows.append(seeded)
            matches = [seeded]

        row = matches[0]
        row["metric_value"] = metric_value
        row["status"] = "measured"
        row["source"] = (
            "nsight_compute_csv_import"
            f"|source_tool={source_tool}"
            f"|source_file={source_file}"
            f"|metric_set={metric_set}"
            f"|import_timestamp={imported_at}"
        )
        imported += 1

    write_csv(
        provenance_path,
        rows=prov_rows,
        fieldnames=[
            "run_id",
            "kernel",
            "problem_size",
            "block_size",
            "metric_name",
            "metric_value",
            "status",
            "source",
        ],
    )

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_nsight = manifest.get("nsight_compute", {})
    previous_imports: list[dict[str, Any]] = []
    if isinstance(previous_nsight, dict):
        if isinstance(previous_nsight.get("imports"), list):
            previous_imports = list(previous_nsight["imports"])

    current_import = {
        "source_csv": str(ncu_csv.resolve()),
        "source_tool": source_tool,
        "source_file": source_file,
        "metric_set": metric_set,
        "import_timestamp": imported_at,
        "imported_metric_rows": imported,
    }
    imports_by_source: dict[str, dict[str, Any]] = {}
    for entry in previous_imports:
        if isinstance(entry, dict):
            key = str(entry.get("source_file", "")).strip()
            if key:
                imports_by_source[key] = entry
    imports_by_source[source_file] = current_import

    measured_profiler_total = sum(
        1
        for row in prov_rows
        if row.get("metric_name") in PROFILER_METRICS and row.get("status") == "measured"
    )
    manifest["nsight_compute"] = {
        "available": True,
        "enabled": True,
        "status": "parsed_metrics",
        "source_csv": str(ncu_csv.resolve()),
        "source_tool": source_tool,
        "source_file": source_file,
        "metric_set": metric_set,
        "import_timestamp": imported_at,
        "imported_metric_rows": imported,
        "imported_metric_rows_total": measured_profiler_total,
        "imports": list(imports_by_source.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    copy_path = run_dir / "ncu_metrics.csv"
    write_csv(
        copy_path,
        rows=ncu_rows,
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )
    return imported
