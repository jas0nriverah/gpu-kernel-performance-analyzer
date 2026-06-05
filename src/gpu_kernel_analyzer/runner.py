from __future__ import annotations

import json
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


def run_binary_for_scenario(binary: Path, scenario: Scenario, interpreter: str | None = None) -> BinaryResult:
    if not binary.exists():
        raise FileNotFoundError(f"Benchmark binary not found: {binary}")

    cmd = _build_command(binary=binary, scenario=scenario, interpreter=interpreter)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark binary failed for kernel={scenario.kernel}, size={scenario.problem_size}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("Benchmark binary produced empty output.")

    raw = json.loads(stdout_lines[-1])
    if not isinstance(raw, dict):
        raise RuntimeError("Benchmark binary JSON output must be an object.")
    return BinaryResult(raw=raw)
