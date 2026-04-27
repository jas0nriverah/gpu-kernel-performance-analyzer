from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional import guard
    yaml = None


SUPPORTED_KERNELS = {"vector_add", "reduction", "gemm_naive", "gemm_tiled"}


@dataclass(frozen=True)
class Scenario:
    kernel: str
    problem_size: int
    block_size: int
    warmups: int
    repeats: int
    verify: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_expand_scenarios(path: Path) -> list[Scenario]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML scenario files.")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    _require(isinstance(payload, dict), "Scenario file must be a JSON object.")
    sweeps = payload.get("sweeps")
    _require(isinstance(sweeps, list) and len(sweeps) > 0, "Scenario file must include non-empty 'sweeps'.")

    expanded: list[Scenario] = []
    for sweep in sweeps:
        _require(isinstance(sweep, dict), "Each sweep must be an object.")
        kernel = sweep.get("kernel")
        _require(kernel in SUPPORTED_KERNELS, f"Unsupported kernel in sweep: {kernel}")

        problem_sizes = sweep.get("problem_sizes")
        block_sizes = sweep.get("block_sizes")
        _require(isinstance(problem_sizes, list) and problem_sizes, "Each sweep needs non-empty 'problem_sizes'.")
        _require(isinstance(block_sizes, list) and block_sizes, "Each sweep needs non-empty 'block_sizes'.")

        warmups = int(sweep.get("warmups", 10))
        repeats = int(sweep.get("repeats", 30))
        verify = bool(sweep.get("verify", False))
        _require(warmups >= 0, "warmups must be >= 0")
        _require(repeats > 0, "repeats must be > 0")

        for problem_size, block_size in itertools.product(problem_sizes, block_sizes):
            ps = int(problem_size)
            bs = int(block_size)
            _require(ps > 0, "problem_size must be > 0")
            _require(bs > 0, "block_size must be > 0")
            expanded.append(
                Scenario(
                    kernel=kernel,
                    problem_size=ps,
                    block_size=bs,
                    warmups=warmups,
                    repeats=repeats,
                    verify=verify,
                )
            )

    if not expanded:
        raise ValueError("No expanded scenarios found.")
    return expanded
