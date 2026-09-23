"""Sustained CUDA workloads with independently sampled NVIDIA board telemetry."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import sha256_file
from .runner import _build_command, validate_binary_output
from .scenarios import load_and_expand_scenarios

QUERY = ["uuid", "name", "power.draw", "utilization.gpu", "utilization.memory",
         "clocks.sm", "clocks.mem", "temperature.gpu"]
COLUMNS = ["gpu_uuid", "gpu_name", "power_watts", "gpu_utilization_pct", "memory_utilization_pct",
           "graphics_clock_mhz", "memory_clock_mhz", "temperature_c"]
NUMERIC = COLUMNS[2:]
SAMPLE_COLUMNS = ["timestamp", "epoch_s", "monotonic_s", "query_duration_s", *COLUMNS,
                  "run_id", "session_id", "scenario_id", "trial", "workload_type", "phase",
                  "backend", "power_scope"]


def query_gpu(device: str) -> dict:
    begin = time.time()
    mono = time.monotonic()
    result = subprocess.run(["nvidia-smi", "-i", device, "--query-gpu=" + ",".join(QUERY),
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=5)
    finish = time.time()
    records = list(csv.reader(result.stdout.strip().splitlines()))
    if len(records) != 1 or len(records[0]) != len(COLUMNS):
        raise ValueError("Power capture requires exactly one NVIDIA GPU and a complete telemetry row")
    row = dict(zip(COLUMNS, (s.strip() for s in records[0]), strict=True))
    for name in NUMERIC:
        try:
            value = float(row[name])
        except ValueError:
            value = None
        row[name] = value if value is not None and math.isfinite(value) else None
    if row["power_watts"] is None or row["power_watts"] <= 0:
        raise ValueError("GPU does not expose a valid positive power reading")
    row.update(epoch_s=(begin + finish) / 2, monotonic_s=mono + (time.monotonic() - mono) / 2,
               query_duration_s=finish - begin,
               timestamp=datetime.fromtimestamp((begin + finish) / 2, timezone.utc).isoformat())
    return row


def summarize_power(samples: list[dict], start: float, end: float, max_gap: float = 1.5) -> dict:
    """Integrate only covered samples; never extrapolate into unobserved boundaries."""
    rows = [r for r in samples if start <= r["epoch_s"] <= end]
    if len(rows) < 3:
        raise ValueError("Insufficient power samples in the measurement window")
    gaps = [b["epoch_s"] - a["epoch_s"] for a, b in zip(rows, rows[1:], strict=False)]
    if any(g <= 0 or g > max_gap for g in gaps):
        raise ValueError("Power trace contains reversed timestamps or excessive sampling gaps")
    if rows[0]["epoch_s"] - start > max_gap or end - rows[-1]["epoch_s"] > max_gap:
        raise ValueError("Power trace does not cover the measurement window")
    for a, b, gap in zip(rows, rows[1:], gaps, strict=False):
        if abs((b["monotonic_s"] - a["monotonic_s"]) - gap) > 0.1:
            raise ValueError("Wall clock changed during power capture")
    watts = [r["power_watts"] for r in rows]
    if any(not isinstance(w, (float, int)) or not math.isfinite(w) or w <= 0 for w in watts):
        raise ValueError("Invalid power samples")
    span = rows[-1]["epoch_s"] - rows[0]["epoch_s"]
    energy = sum((a + b) * 0.5 * dt for a, b, dt in zip(watts, watts[1:], gaps, strict=False))
    return dict(sample_count=len(rows), sample_span_s=span, coverage_fraction=span / (end - start),
                first_sample_epoch_s=rows[0]["epoch_s"], last_sample_epoch_s=rows[-1]["epoch_s"],
                mean_power_w=energy / span, min_power_w=min(watts), max_power_w=max(watts),
                estimated_energy_j=energy, max_sample_gap_s=max(gaps),
                max_query_duration_s=max(r["query_duration_s"] for r in rows))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def capture_power(args) -> int:
    for name in ("duration_seconds", "interval_seconds", "idle_seconds", "trim_seconds", "timeout_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.duration_seconds < 2 * args.trim_seconds + 5 or args.idle_seconds < 3:
        raise ValueError("Use at least five measured seconds after trimming, and at least three idle seconds")
    if args.interval_seconds > 1 or args.trials < 1:
        raise ValueError("Sampling interval must be <= 1 second; trials must be positive")
    if args.timeout_seconds <= args.duration_seconds + 5:
        raise ValueError("Worker timeout must exceed sustained duration by more than five seconds")
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    scenarios = load_and_expand_scenarios(Path(args.scenarios))
    if not scenarios or any(not s.verify for s in scenarios):
        raise ValueError("Power capture requires scenarios with correctness verification enabled")
    device = query_gpu(args.device)
    device_uuid = device["gpu_uuid"]
    # UUID selection gives CUDA and telemetry the same physical device, including under Slurm.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (args.device, device_uuid, "0"):
        raise ValueError("Ambiguous CUDA_VISIBLE_DEVICES mapping; select the allocated GPU UUID explicitly")
    processes = subprocess.run(["nvidia-smi", "-i", device_uuid,
                                "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True, timeout=5).stdout
    if processes.strip():
        raise ValueError("GPU already has compute processes; use an idle allocated GPU")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "workers").mkdir()
    run_id = "power_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    manifest = dict(schema_version=1, status="running", run_id=run_id, device=device,
                    source_commit=_git(["rev-parse", "HEAD"]), source_dirty=bool(_git(["status", "--porcelain"])),
                    binary_sha256=sha256_file(binary), scenario_sha256=sha256_file(Path(args.scenarios)),
                    protocol={k: getattr(args, k) for k in ("duration_seconds", "interval_seconds", "idle_seconds",
                              "trim_seconds", "trials", "seed", "timeout_seconds")},
                    backend="cuda", power_source="nvidia-smi:power.draw", power_scope="gpu_board",
                    energy_status="derived_trapezoidal_integration", original_cuda_visible_devices=visible,
                    note="Board power includes associated circuitry. Sensor averaging is device dependent; H100 reports approximately one-second averaged power. No clock or power-limit changes.")
    source_files = [*Path("src").rglob("*.py"), *Path("benchmarks").rglob("*.cu"), *Path("benchmarks/include").glob("*.h"), Path("benchmarks/CMakeLists.txt"), Path("pyproject.toml")]
    manifest["source_files_sha256"] = {str(p): sha256_file(p) for p in source_files}
    manifest["environment"] = subprocess.check_output(["nvidia-smi", "-i", device_uuid,
        "--query-gpu=uuid,name,driver_version,power.limit,clocks.sm,clocks.mem,mig.mode.current", "--format=csv"], text=True, timeout=5)
    _write_json(out / "manifest.json", manifest)
    samples: list[dict] = []
    errors: list[Exception] = []
    stop = threading.Event()
    context = dict(run_id=run_id, session_id="", scenario_id="", trial=0, workload_type="idle",
                   phase="idle", backend="cuda", power_scope="gpu_board")
    lock = threading.Lock()

    def sample():
        try:
            with (out / "telemetry.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
                writer.writeheader()
                while not stop.is_set():
                    with lock:
                        labels = context.copy()
                    row = query_gpu(device_uuid)
                    if row["gpu_uuid"] != device_uuid:
                        raise ValueError("Telemetry device identity changed")
                    row.update(labels)
                    writer.writerow(row)
                    handle.flush()
                    samples.append(row)
                    stop.wait(args.interval_seconds)
        except Exception as exc:
            errors.append(exc)
            stop.set()

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    summaries = []
    training_rows = []
    try:
        rng = random.Random(args.seed)
        for trial in range(1, args.trials + 1):
            ordered = list(scenarios)
            rng.shuffle(ordered)
            for scenario in ordered:
                key = f"{scenario.kernel}-{scenario.problem_size}-{scenario.block_size}"
                with lock:
                    context.update(session_id=f"{run_id}_trial_{trial}", scenario_id=key, trial=trial,
                                   workload_type=scenario.kernel, phase="idle")
                idle_start = time.time()
                if stop.wait(args.idle_seconds):
                    raise RuntimeError(f"Telemetry failed: {errors}")
                idle_end = time.time()
                idle = summarize_power(samples, idle_start + 1, idle_end)
                with lock:
                    context["phase"] = "workload"
                command = _build_command(binary, scenario, None) + ["--sustain-seconds", str(args.duration_seconds)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device_uuid)
                proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout_seconds)
                worker = out / "workers" / f"trial-{trial}-{key}"
                worker.with_suffix(".json").write_text(proc.stdout)
                worker.with_suffix(".stderr").write_text(proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f"Worker failed: {proc.stderr}")
                raw = json.loads(proc.stdout.strip().splitlines()[-1])
                validate_binary_output(raw, scenario)
                markers = [json.loads(line.removeprefix("POWER_WINDOW ")) for line in proc.stderr.splitlines()
                           if line.startswith("POWER_WINDOW ")]
                if len(markers) != 1:
                    raise ValueError("Worker must emit exactly one sustained measurement window")
                window = markers[0]
                if window["launches"] <= 0 or window["duration_s"] < args.duration_seconds:
                    raise ValueError("Invalid sustained workload duration or launch count")
                if abs(window["end_epoch_s"] - window["start_epoch_s"] - window["duration_s"]) > 0.1:
                    raise ValueError("Worker clock changed during measurement")
                if errors:
                    raise RuntimeError(f"Telemetry failed: {errors}")
                start = window["start_epoch_s"] + args.trim_seconds
                end = window["end_epoch_s"] - args.trim_seconds
                stats = summarize_power(samples, start, end)
                launches_per_second = window["launches"] / window["duration_s"]
                summary = dict(trial=trial, scenario_id=key, kernel=scenario.kernel,
                               problem_size=scenario.problem_size, block_size=scenario.block_size,
                               **stats, idle_mean_power_w=idle["mean_power_w"],
                               sustained_launches=window["launches"], sustained_duration_s=window["duration_s"],
                               launches_per_second=launches_per_second,
                               estimated_j_per_launch=stats["mean_power_w"] / launches_per_second,
                               effective_gflops=raw["flops"] * launches_per_second / 1e9,
                               effective_gbps=raw["bytes_moved"] * launches_per_second / 1e9,
                               event_runtime_ms_median=statistics.median(raw["runtime_ms_samples"]),
                               verification_passed=True, window=window)
                summaries.append(summary)
                training_rows.extend(dict(r, phase="steady") for r in samples if start <= r["epoch_s"] <= end)
                _write_json(out / "summary.json", summaries)
                print(f"Trial {trial}: {key}: {stats['mean_power_w']:.1f} W, {stats['sample_count']} samples, verified", flush=True)
        stop.set()
        thread.join(timeout=7)
        if thread.is_alive() or errors:
            raise RuntimeError(f"Telemetry did not finish cleanly: {errors}")
        with (out / "steady_telemetry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(training_rows)
        fields = [k for k in summaries[0] if k != "window"]
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        telemetry_samples=len(samples), steady_samples=len(training_rows),
                        artifacts_sha256={str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"})
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        stop.set()
        thread.join(timeout=7)
        _write_json(out / "manifest.json", manifest)
    return 0


def add_power_parser(sub):
    power = sub.add_parser("power", help="Capture measured board power during sustained CUDA workloads.")
    commands = power.add_subparsers(dest="power_command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--binary", required=True)
    capture.add_argument("--scenarios", required=True)
    capture.add_argument("--outdir", required=True)
    capture.add_argument("--device", default="0", help="Allocated NVIDIA device UUID or nvidia-smi index.")
    capture.add_argument("--duration-seconds", type=float, default=15)
    capture.add_argument("--interval-seconds", type=float, default=0.2)
    capture.add_argument("--idle-seconds", type=float, default=4)
    capture.add_argument("--trim-seconds", type=float, default=2)
    capture.add_argument("--timeout-seconds", type=float, default=120)
    capture.add_argument("--trials", type=int, default=3)
    capture.add_argument("--seed", type=int, default=42)
    capture.set_defaults(func=capture_power)
