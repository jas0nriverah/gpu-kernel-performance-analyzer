"""Measure an already-running API over HTTP; this is a local microbenchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from gpu_power_pipeline.data import load_dataset


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--requests", type=positive_int, default=100)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("--url must identify a localhost HTTP(S) server")
    if args.outdir.exists() and (not args.outdir.is_dir() or any(args.outdir.iterdir())):
        parser.error("--outdir must be new or empty")
    args.outdir.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/")
    with httpx.Client(
        timeout=60,
        trust_env=False,
        limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency),
    ) as client:
        metadata_response = client.get(url + "/model")
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        frame = load_dataset("local_nvidia_smi", data_path=str(args.telemetry))
        required_fields = metadata.get("required_fields", metadata["feature_order"])
        missing = set(required_fields) - set(frame.columns)
        if missing:
            raise ValueError(f"Telemetry lacks required serving fields: {sorted(missing)}")
        records = json.loads(frame[required_fields].to_json(orient="records"))
        if not records:
            raise ValueError("Telemetry has no records")

        def request(index: int) -> dict:
            batch = [records[(index * args.batch_size + offset) % len(records)] for offset in range(args.batch_size)]
            start = time.perf_counter()
            status = None
            error = ""
            try:
                response = client.post(url + "/predict", json={"records": batch})
                status = response.status_code
                response.raise_for_status()
                body = response.json()
                if (body["model_name"], body["model_version"]) != (metadata["model_name"], metadata["version"]):
                    raise ValueError("Response model identity changed")
                predictions = body["predictions"]
                if len(predictions) != len(batch):
                    raise ValueError("Response row count mismatch")
                if any(
                    item["index"] != i or not math.isfinite(item["predicted_power_watts"])
                    for i, item in enumerate(predictions)
                ):
                    raise ValueError("Response contains invalid predictions")
            except Exception as exc:  # noqa: BLE001 - preserve failed request evidence
                error = str(exc)
            return {
                "request": index,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "status": status,
                "rows": len(batch),
                "success": not error,
                "error": error,
            }

        warmups = [request(i) for i in range(args.warmups)]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            # map consumes only a bounded number of requests in flight: one per worker.
            results = list(pool.map(request, range(args.requests)))
        elapsed = time.perf_counter() - start

    def percentiles(rows: list[dict]) -> dict:
        if not rows:
            return {"p50": None, "p95": None, "p99": None}
        values = np.percentile([row["latency_ms"] for row in rows], [50, 95, 99])
        return dict(zip(["p50", "p95", "p99"], map(float, values), strict=True))

    successes = [row for row in results if row["success"]]
    errors = len(results) - len(successes)
    warmup_errors = sum(not row["success"] for row in warmups)
    summary = {
        "scope": "Local HTTP microbenchmark of one running API; not production capacity.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "telemetry": str(args.telemetry.resolve()),
        "model_name": metadata["model_name"],
        "model_version": metadata["version"],
        "feature_order": metadata["feature_order"],
        "required_fields": required_fields,
        "environment": {"python": sys.version, "platform": platform.platform(), "httpx": httpx.__version__},
        "requests": args.requests,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "warmup_errors": warmup_errors,
        "elapsed_wall_seconds": elapsed,
        "successful_requests": len(successes),
        "error_count": errors,
        "successful_records": sum(row["rows"] for row in successes),
        "successful_requests_per_second": len(successes) / elapsed,
        "successful_records_per_second": sum(row["rows"] for row in successes) / elapsed,
        "latency_ms_all_requests": percentiles(results),
        "latency_ms_successful_requests": percentiles(successes),
    }
    for name, rows in [("requests.csv", results), ("warmups.csv", warmups)]:
        with (args.outdir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["request", "latency_ms", "status", "rows", "success", "error"])
            writer.writeheader()
            writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(bool(errors or warmup_errors))


if __name__ == "__main__":
    raise SystemExit(main())
