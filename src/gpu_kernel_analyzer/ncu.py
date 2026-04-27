from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from .io import read_csv, write_csv
from .schemas import PROFILER_METRICS


ALLOWED_SOURCE_TOOLS = {"ncu", "nsight_compute"}


def _required_text(value: str, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"Missing required provenance field: {field}")
    return text


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
    kernel,metric_name,metric_value
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

        matches = [
            row
            for row in prov_rows
            if row.get("kernel") == kernel
            and row.get("problem_size") == problem_size
            and row.get("block_size") == block_size
            and row.get("metric_name") == metric_name
        ]
        if not matches:
            raise RuntimeError(
                "No matching benchmark scenario for Nsight metric row: "
                f"kernel={kernel}, problem_size={problem_size}, block_size={block_size}, metric={metric_name}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                "Ambiguous Nsight metric mapping; multiple matching benchmark scenarios found for: "
                f"kernel={kernel}, problem_size={problem_size}, block_size={block_size}, metric={metric_name}"
            )

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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    copy_path = run_dir / "ncu_metrics.csv"
    write_csv(
        copy_path,
        rows=ncu_rows,
        fieldnames=["kernel", "problem_size", "block_size", "metric_name", "metric_value"],
    )
    return imported
