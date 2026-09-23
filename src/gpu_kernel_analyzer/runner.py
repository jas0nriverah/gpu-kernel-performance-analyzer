from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scenarios import Scenario


@dataclass
class BinaryResult:
    raw: dict[str, Any]

    @property
    def runtime_ms_samples(self) -> list[float]:
        values = self.raw.get("runtime_ms_samples", [])
        return [float(v) for v in values]


def _build_command(binary: Path, scenario: Scenario, interpreter: str | None) -> list[str]:
    base: list[str] = []
    if interpreter:
        base.append(interpreter)
    base.append(str(binary))
    base.extend(
        [
            "--kernel",
            scenario.kernel,
            "--problem-size",
            str(scenario.problem_size),
            "--block-size",
            str(scenario.block_size),
            "--warmups",
            str(scenario.warmups),
            "--repeats",
            str(scenario.repeats),
        ]
    )
    if scenario.verify:
        base.append("--verify")
    return base


def run_binary_for_scenario(
    binary: Path, scenario: Scenario, interpreter: str | None = None, timeout_seconds: float = 120.0,
) -> BinaryResult:
    if not binary.exists():
        raise FileNotFoundError(f"Benchmark binary not found: {binary}")

    cmd = _build_command(binary=binary, scenario=scenario, interpreter=interpreter)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Benchmark timed out after {timeout_seconds}s: kernel={scenario.kernel}, "
            f"size={scenario.problem_size}, block={scenario.block_size}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark binary failed for kernel={scenario.kernel}, size={scenario.problem_size}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("Benchmark binary produced empty output.")

    raw = json.loads(stdout_lines[-1])
    validate_binary_output(raw, scenario)
    return BinaryResult(raw=raw)


def validate_binary_output(raw: dict, scenario: Scenario) -> None:
    if not isinstance(raw, dict):
        raise RuntimeError("Benchmark binary JSON output must be an object.")
    for field in ("kernel", "problem_size", "block_size", "warmups", "repeats", "verify"):
        if raw.get(field) != getattr(scenario, field):
            raise RuntimeError(f"Benchmark output does not match requested {field}: {raw.get(field)!r}")
    if scenario.verify and raw.get("verification_passed") is not True:
        raise RuntimeError(f"Benchmark correctness verification failed or missing: {scenario.kernel}")
    for field in ("bytes_moved", "flops"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Benchmark {field} must be a finite nonnegative number")
    samples = raw.get("runtime_ms_samples")
    if not isinstance(samples, list) or len(samples) != scenario.repeats:
        raise RuntimeError("Benchmark sample count does not match requested repeats")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples):
        raise RuntimeError("Benchmark timings must be finite positive numbers")
