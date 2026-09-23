"""Measured, correctness-gated kernel configuration search.

The tuner deliberately searches only existing kernel families and launch block
sizes. It ranks captured measurements, never performance-model predictions.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import subprocess
from argparse import Namespace
from pathlib import Path

from .artifacts import sha256_file
from .power import capture_power, query_gpu
from .runner import _build_command, validate_binary_output
from .scenarios import FIXED_BLOCK_SIZE_KERNELS, SUPPORTED_KERNELS, Scenario


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def workload_key(kernel: str, size: int) -> str:
    # These GEMM implementations compute the same mathematical workload.
    family = "gemm" if kernel in {"gemm_naive", "gemm_tiled"} else kernel
    return f"{family}:{size}"


def validate_capture(path: Path, expected: dict) -> list[dict]:
    """Load complete, untampered capture evidence and validate trial coverage."""
    manifest_path = path / "manifest.json"
    summary_path = path / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Capture is incomplete: {path}")
    if manifest.get("backend") != "cuda" or manifest.get("power_scope") != "gpu_board":
        raise ValueError("Capture backend or power scope is incompatible")
    if manifest.get("binary_sha256") != expected["binary_sha256"]:
        raise ValueError("Capture binary identity does not match")
    if manifest.get("device", {}).get("gpu_uuid") != expected["gpu_uuid"]:
        raise ValueError("Capture GPU identity does not match")
    for name, value in expected["protocol"].items():
        if manifest.get("protocol", {}).get(name) != value:
            raise ValueError(f"Capture protocol does not match {name}")
    hashes = manifest.get("artifacts_sha256")
    required = {"summary.json", "summary.csv", "telemetry.csv", "steady_telemetry.csv"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("Capture artifact hash coverage is incomplete")
    worker_files = sorted((path / "workers").glob("trial-*.json"))
    if len(worker_files) != expected["trials"] or not {str(p.relative_to(path)) for p in worker_files}.issubset(hashes):
        raise ValueError("Capture worker artifact hashes are incomplete")
    for rel, digest in hashes.items():
        artifact = path / rel
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"Capture artifact is missing or tampered: {rel}")
    rows = json.loads(summary_path.read_text())
    if not isinstance(rows, list) or len(rows) != expected["trials"]:
        raise ValueError("Capture does not contain the requested number of trials")
    trials = set()
    workers = worker_files
    if len(workers) != expected["trials"]:
        raise ValueError("Capture worker evidence does not cover every trial")
    for worker_file in workers:
        worker = json.loads(worker_file.read_text())
        worker_device = worker.get("device", {}).get("uuid")
        if worker_device != expected["gpu_uuid"]:
            raise ValueError("Capture worker GPU identity does not match telemetry GPU")
        worker_trial = int(worker_file.name.split("-", 2)[1])
        scenario = Scenario(expected["kernel"], expected["problem_size"], expected["block_size"],
                            expected["warmups"], expected["repeats"], True)
        validate_binary_output(worker, scenario)
        if worker.get("verification_passed") is not True:
            raise ValueError("Capture worker correctness verification failed")
        if worker_trial not in range(1, expected["trials"] + 1):
            raise ValueError("Capture worker trial identity is invalid")
    for row in rows:
        if (row.get("kernel"), row.get("problem_size"), row.get("block_size")) != (
            expected["kernel"], expected["problem_size"], expected["block_size"]
        ):
            raise ValueError("Capture contains a different candidate")
        trial = row.get("trial")
        if isinstance(trial, bool) or not isinstance(trial, int) or trial in trials:
            raise ValueError("Capture has invalid or duplicate trial coverage")
        trials.add(trial)
        if row.get("verification_passed") is not True:
            raise ValueError("Measured candidate failed correctness verification")
        for field in ("estimated_j_per_launch", "sustained_duration_s", "launches_per_second", "mean_power_w"):
            if _finite(row.get(field), field) <= 0:
                raise ValueError(f"Capture has invalid {field}")
        window = row.get("window", {})
        if isinstance(window.get("launches"), bool) or not isinstance(window.get("launches"), int) or window["launches"] <= 0:
            raise ValueError("Capture has invalid launch count")
        expected_rate = window["launches"] / row["sustained_duration_s"]
        if not math.isclose(row["launches_per_second"], expected_rate, rel_tol=1e-5):
            raise ValueError("Capture launch rate is inconsistent with its worker window")
        if not math.isclose(row["estimated_j_per_launch"], row["mean_power_w"] / expected_rate, rel_tol=1e-5):
            raise ValueError("Capture energy is inconsistent with measured power and launch rate")
    if trials != set(range(1, expected["trials"] + 1)):
        raise ValueError("Capture is missing trial coverage")
    return rows


def summarize_candidate(expected: dict, rows: list[dict]) -> dict:
    times = [r["sustained_duration_s"] / r["window"]["launches"] for r in rows]
    energies = [r["estimated_j_per_launch"] for r in rows]
    return {**expected, "valid": True, "trials": len(rows),
            "runtime_s_per_launch_median": statistics.median(times),
            "runtime_s_per_launch_mad": statistics.median(abs(x - statistics.median(times)) for x in times),
            "energy_j_per_launch_median": statistics.median(energies),
            "energy_j_per_launch_mad": statistics.median(abs(x - statistics.median(energies)) for x in energies),
            "raw_trials": rows}


def choose_candidates(candidates: list[dict], speed_tolerance: float = 0.05) -> dict:
    """Select minimum-energy measured candidate within tolerance of fastest."""
    tolerance = _finite(speed_tolerance, "speed_tolerance")
    if tolerance < 0:
        raise ValueError("speed_tolerance must be nonnegative")
    groups: dict[str, list[dict]] = {}
    for c in candidates:
        if not c.get("valid"):
            continue
        if c["workload"] != workload_key(c["kernel"], c["problem_size"]):
            raise ValueError("Candidate workload label is incompatible")
        for f in ("runtime_s_per_launch_median", "energy_j_per_launch_median"):
            if _finite(c.get(f), f) <= 0:
                raise ValueError(f"Candidate has invalid {f}")
        groups.setdefault(c["workload"], []).append(c)
    result = {}
    for key, group in groups.items():
        fastest = min(c["runtime_s_per_launch_median"] for c in group)
        eligible = [c for c in group if c["runtime_s_per_launch_median"] <= fastest * (1 + tolerance)]
        selected = min(eligible, key=lambda c: (c["energy_j_per_launch_median"],
                                                 c["runtime_s_per_launch_median"], c["candidate_id"]))
        frontier = [c for c in group if not any(
            d is not c and d["runtime_s_per_launch_median"] <= c["runtime_s_per_launch_median"]
            and d["energy_j_per_launch_median"] <= c["energy_j_per_launch_median"]
            and (d["runtime_s_per_launch_median"] < c["runtime_s_per_launch_median"]
                 or d["energy_j_per_launch_median"] < c["energy_j_per_launch_median"])
            for d in group)]
        result[key] = {"fastest_runtime_s_per_launch": fastest,
                       "speed_eligible": [c["candidate_id"] for c in eligible],
                       "pareto_frontier": [c["candidate_id"] for c in frontier],
                       "selected_candidate_id": selected["candidate_id"]}
    return result


def compare_confirmed_energy(measured: list[dict], confirmed: list[dict], eligible_ids: list[str],
                             selected_id: str) -> dict:
    """Compare the selected energy against independently repeated eligible candidates."""
    measured_ids = {c["candidate_id"] for c in measured}
    confirmed_by_id = {c["candidate_id"]: c for c in confirmed if c.get("confirmation_passed")}
    order = sorted((confirmed_by_id[cid] for cid in eligible_ids if cid in confirmed_by_id),
                   key=lambda c: c["energy_j_per_launch_median"])
    lowest_id = order[0]["candidate_id"] if order else None
    tied_ids = []
    if order:
        low = order[0]
        for other in order[1:]:
            low_initial = next(c for c in measured if c["candidate_id"] == lowest_id)
            other_initial = next(c for c in measured if c["candidate_id"] == other["candidate_id"])
            discovery_gap = abs(low_initial["energy_j_per_launch_median"] - other_initial["energy_j_per_launch_median"])
            discovery_dispersion = low_initial["energy_j_per_launch_mad"] + other_initial["energy_j_per_launch_mad"]
            confirm_gap = abs(low["energy_j_per_launch_median"] - other["energy_j_per_launch_median"])
            confirm_dispersion = low["energy_j_per_launch_mad"] + other["energy_j_per_launch_mad"]
            if discovery_gap <= discovery_dispersion or confirm_gap <= confirm_dispersion:
                tied_ids.append(other["candidate_id"])
    unique_minimum = bool(set(eligible_ids).issubset(measured_ids) and len(order) == len(eligible_ids)
                          and lowest_id == selected_id and not tied_ids)
    tied_set = {lowest_id, *tied_ids} if tied_ids else set()
    fastest_tied = min((c for c in order if c["candidate_id"] in tied_set),
                       key=lambda c: c["runtime_s_per_launch_median"], default=None)
    return {
        "confirmed_lowest_energy_candidate_id": lowest_id,
        "indistinguishable_energy_candidates": sorted(tied_set),
        "fastest_practical_recommendation_if_tied": fastest_tied["candidate_id"] if fastest_tied else None,
        "unique_energy_minimum_confirmed": unique_minimum,
        "method": "Candidates are treated as indistinguishable if their discovery or confirmation energy median gap is no larger than the sum of the corresponding candidates' median absolute deviations for that phase. This is a descriptive dispersion screen, not a significance test.",
    }


def _capture(binary: Path, scenario: Scenario, out: Path, device: str, args, seed: int, trials: int) -> list[dict]:
    out.mkdir(parents=True, exist_ok=False)
    config = out / "scenario.json"
    config.write_text(json.dumps({"sweeps": [{"kernel": scenario.kernel,
        "problem_sizes": [scenario.problem_size], "block_sizes": [scenario.block_size],
        "warmups": scenario.warmups, "repeats": scenario.repeats, "verify": True}]}))
    capture_power(Namespace(binary=str(binary), scenarios=str(config), outdir=str(out / "capture"), device=device,
        duration_seconds=args.duration_seconds, interval_seconds=args.interval_seconds,
        idle_seconds=args.idle_seconds, trim_seconds=args.trim_seconds,
        timeout_seconds=args.timeout_seconds, trials=trials, seed=seed))
    expected = {"kernel": scenario.kernel, "problem_size": scenario.problem_size,
                "block_size": scenario.block_size, "trials": trials, "binary_sha256": sha256_file(binary),
                "gpu_uuid": device, "warmups": scenario.warmups, "repeats": scenario.repeats,
                "protocol": {"trials": trials, "seed": seed,
                "duration_seconds": args.duration_seconds, "interval_seconds": args.interval_seconds,
                "idle_seconds": args.idle_seconds, "trim_seconds": args.trim_seconds,
                "timeout_seconds": args.timeout_seconds}}
    return validate_capture(out / "capture", expected)


def run_autotune(args) -> int:
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not args.kernels or set(args.kernels) - SUPPORTED_KERNELS:
        raise ValueError("Specify supported kernels")
    if min(args.trials, args.confirm_trials) < 2:
        raise ValueError("Measurement and confirmation each need at least two trials")
    if not (0 <= args.speed_tolerance <= 0.5):
        raise ValueError("speed-tolerance must be between 0 and 0.5")
    sizes = sorted(set(args.problem_sizes))
    blocks = sorted(set(args.block_sizes))
    if not sizes or not blocks or any(isinstance(x, bool) or x <= 0 for x in sizes + blocks):
        raise ValueError("Problem sizes and block sizes must be positive")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    device_info = query_gpu(args.device)
    device = device_info["gpu_uuid"]
    candidates = []
    failures = []
    configs = [(k, s, b) for k in args.kernels for s in sizes for b in blocks
               if k not in FIXED_BLOCK_SIZE_KERNELS or b == FIXED_BLOCK_SIZE_KERNELS[k]]
    if not configs:
        raise ValueError("No requested block sizes are valid for the selected kernels")
    rng = random.Random(args.seed)
    rng.shuffle(configs)
    for index, (kernel, size, block) in enumerate(configs):
        cid = f"{kernel}-n{size}-b{block}"
        scenario = Scenario(kernel, size, block, args.warmups, args.repeats, True)
        item = {"candidate_id": cid, "kernel": kernel, "problem_size": size, "block_size": block,
                "workload": workload_key(kernel, size)}
        candidate_dir = out / "candidates" / cid
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Explicit visibility ensures correctness runs target the same device as telemetry.
            command = _build_command(binary, scenario, None)
            proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout_seconds,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=device))
            (candidate_dir / "preflight.stdout").write_text(proc.stdout)
            (candidate_dir / "preflight.stderr").write_text(proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"Preflight failed: {proc.stderr[-1000:]}")
            raw = json.loads(proc.stdout.strip().splitlines()[-1])
            validate_binary_output(raw, scenario)
            if raw.get("verification_passed") is not True:
                raise ValueError("Correctness preflight did not pass")
            if raw.get("device", {}).get("uuid") != device:
                raise ValueError("Correctness preflight ran on a different GPU")
            rows = _capture(binary, scenario, candidate_dir / "measurement", device, args,
                            args.seed + index * 1009, args.trials)
            candidates.append(summarize_candidate(item, rows))
        except Exception as exc:
            if not (candidate_dir / "preflight.stderr").exists():
                (candidate_dir / "preflight.stderr").write_text(str(exc) + "\n")
            failures.append({**item, "valid": False, "reason": str(exc)})
    selection = choose_candidates(candidates, args.speed_tolerance)
    requested_workloads = {workload_key(k, s) for k, s, _ in configs}
    missing_workloads = sorted(requested_workloads - set(selection))
    # Recapture all speed-eligible candidates so energy selection is confirmed too.
    by_id = {c["candidate_id"]: c for c in candidates}
    fastest_ids = {min((c for c in candidates if c["workload"] == key),
                       key=lambda c: c["runtime_s_per_launch_median"])["candidate_id"] for key in selection}
    finalists = sorted({i for s in selection.values() for i in s["speed_eligible"]} | fastest_ids
                       | {s["selected_candidate_id"] for s in selection.values()})
    rng.shuffle(finalists)
    confirmations = []
    for index, cid in enumerate(finalists):
        c = by_id[cid]
        scenario = Scenario(c["kernel"], c["problem_size"], c["block_size"], args.warmups, args.repeats, True)
        try:
            rows = _capture(binary, scenario, out / "confirmations" / cid, device, args,
                            args.seed + 1_000_003 + index * 7919, args.confirm_trials)
            fresh = summarize_candidate({k: c[k] for k in ("candidate_id", "kernel", "problem_size", "block_size", "workload")}, rows)
            original = c["energy_j_per_launch_median"]
            runtime_original = c["runtime_s_per_launch_median"]
            fresh["confirmation_passed"] = (
                abs(fresh["energy_j_per_launch_median"] - original) <= max(original * 0.20, 1e-12)
                and abs(fresh["runtime_s_per_launch_median"] - runtime_original) <= max(runtime_original * 0.20, 1e-12))
            fresh["comparison_limits"] = "Independent medians must agree within 20%; device sensor averaging and small trial counts remain uncertainty sources."
            confirmations.append(fresh)
        except Exception as exc:
            confirmations.append({"candidate_id": cid, "confirmation_passed": False, "reason": str(exc)})
    confirmed_by_id = {x["candidate_id"]: x for x in confirmations}
    energy_comparison = {}
    for workload, choice in selection.items():
        selected_id = choice["selected_candidate_id"]
        eligible_ids = choice["speed_eligible"]
        confirmed_eligible = [confirmed_by_id.get(cid, {}) for cid in eligible_ids]
        confirmed_eligible = [x for x in confirmed_eligible if x.get("confirmation_passed")]
        fastest_id = min((x["candidate_id"] for x in confirmed_eligible),
                         key=lambda cid: confirmed_by_id[cid]["runtime_s_per_launch_median"], default=None)
        selected_confirmation = confirmed_by_id.get(selected_id, {})
        fastest_confirmation = confirmed_by_id.get(fastest_id, {})
        selected_confirmation["speed_constraint_passed"] = bool(
            selected_confirmation.get("confirmation_passed") and fastest_confirmation.get("confirmation_passed")
            and selected_confirmation["runtime_s_per_launch_median"] <= fastest_confirmation["runtime_s_per_launch_median"] * (1 + args.speed_tolerance))
        measured_group = [c for c in candidates if c["workload"] == workload]
        energy_comparison[workload] = compare_confirmed_energy(
            measured_group, confirmed_eligible, eligible_ids, selected_id)
        selected_confirmation["energy_selection_passed"] = energy_comparison[workload]["unique_energy_minimum_confirmed"]
    selected_confirmed = {x["candidate_id"] for x in confirmations
                          if x.get("confirmation_passed") and x.get("speed_constraint_passed", True)
                          and x.get("energy_selection_passed", True)}
    status = "complete" if selection and not missing_workloads and all(
        x["selected_candidate_id"] in selected_confirmed for x in selection.values()) else (
            "unconfirmed" if selection else "failed")
    report = {"schema_version": 1, "status": status, "device": device,
        "binary_sha256": sha256_file(binary), "protocol": {"trials": args.trials,
        "confirm_trials": args.confirm_trials, "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds, "trim_seconds": args.trim_seconds,
        "seed": args.seed, "confirmation_seed_offset": 1_000_003,
        "speed_tolerance": args.speed_tolerance}, "selection": selection,
        "missing_requested_workloads": missing_workloads,
        "selected_confirmed": {k: s["selected_candidate_id"] in selected_confirmed for k, s in selection.items()},
        "energy_comparison": energy_comparison,
        "candidates": candidates, "disqualified_candidates": failures, "confirmations": confirmations,
        "noise_note": "Median absolute deviation is reported per run. Confirmations use separate capture runs and need 20% agreement; selected candidates must also remain within the speed tolerance of the confirmed fastest candidate and be the unique measured energy minimum outside the dispersion screen. Tied candidates are reported as indistinguishable without a unique energy winner. Device sensor averaging and small trial counts remain uncertainty sources. Candidate captures run sequentially.",
        "pareto_tradeoffs": {key: [by_id[i] for i in value["pareto_frontier"]] for key, value in selection.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(_markdown(report))
    if not selection or missing_workloads or not all(report["selected_confirmed"].values()):
        return 2
    return 0


def _markdown(report: dict) -> str:
    lines = ["# Measured autotuning results", "", "All rankings use measured sustained trials. Runtime/energy are per launch; workloads are grouped by mathematical operation and size.", ""]
    for workload, pick in report["selection"].items():
        lines += [f"## {workload}", "", f"Selected: `{pick['selected_candidate_id']}` (confirmation: **{'pass' if report['selected_confirmed'].get(workload) else 'fail'}**)", "", "| Candidate | Median runtime (s/launch) | Median energy (J/launch) | Runtime MAD | Energy MAD |", "|---|---:|---:|---:|---:|"]
        for c in report["candidates"]:
            if c["workload"] != workload:
                continue
            lines.append(f"| {c['candidate_id']} | {c['runtime_s_per_launch_median']:.6g} | {c['energy_j_per_launch_median']:.6g} | {c['runtime_s_per_launch_mad']:.3g} | {c['energy_j_per_launch_mad']:.3g} |")
        lines.append("")
        energy = report.get("energy_comparison", {}).get(workload, {})
        if energy.get("indistinguishable_energy_candidates"):
            lines.append("Indistinguishable by the MAD dispersion screen: " + ", ".join(f"`{x}`" for x in energy["indistinguishable_energy_candidates"]) + ". No unique energy minimum is claimed.")
            if energy.get("fastest_practical_recommendation_if_tied"):
                lines.append(f"Fastest practical recommendation among that tied set: `{energy['fastest_practical_recommendation_if_tied']}`.")
            lines.append("")
    lines += ["## Evidence and limits", "", "Raw telemetry, trial summaries, worker stdout/stderr, manifests, preflight failures, and independent confirmation captures are retained in this output directory. Capture hashes are checked before ranking. Confirmation compares independent medians within 20%; this is a coarse repeatability screen, not a statistical confidence interval. GPU board power includes device-level circuitry and sensor averaging.", ""]
    return "\n".join(lines)


def add_autotune_parser(sub):
    parser = sub.add_parser("autotune", help="Search existing kernel/block choices using measured time and board energy.")
    commands = parser.add_subparsers(dest="autotune_command", required=True)
    run = commands.add_parser("run", help="Correctness-gate, measure, select, and independently confirm candidates.")
    run.add_argument("--binary", required=True)
    run.add_argument("--kernels", nargs="+", required=True, choices=sorted(SUPPORTED_KERNELS))
    run.add_argument("--problem-sizes", nargs="+", type=int, required=True)
    run.add_argument("--block-sizes", nargs="+", type=int, required=True)
    run.add_argument("--outdir", required=True)
    run.add_argument("--device", default="0")
    run.add_argument("--trials", type=int, default=3)
    run.add_argument("--confirm-trials", type=int, default=3)
    run.add_argument("--duration-seconds", type=float, default=15)
    run.add_argument("--interval-seconds", type=float, default=0.2)
    run.add_argument("--idle-seconds", type=float, default=4)
    run.add_argument("--trim-seconds", type=float, default=2)
    run.add_argument("--timeout-seconds", type=float, default=120)
    run.add_argument("--warmups", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--speed-tolerance", type=float, default=0.05)
    run.set_defaults(func=run_autotune)
