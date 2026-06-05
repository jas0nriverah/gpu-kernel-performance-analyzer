from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional import guard
    yaml = None


SUPPORTED_KERNELS = {
    "vector_add",
    "reduction",
    "gemm_naive",
    "gemm_tiled",
    "memcpy_bandwidth",
    "stencil_1d",
}

# gemm_tiled uses a compile-time 16x16 shared-memory tile, so the launch block
# must be 16 (16x16 threads). Enforce this at config-expansion time to fail fast
# instead of only failing inside the CUDA binary.
FIXED_BLOCK_SIZE_KERNELS = {"gemm_tiled": 16}

# Kernels that read a power-of-two-friendly reduction tree benefit from block
# sizes that are a power of two; flag others as a soft warning.
POWER_OF_TWO_BLOCK_KERNELS = {"reduction"}

# Kernels launched with a 1D block, where block_size is the total thread count.
# (GEMM kernels use a 2D block_size x block_size tile, so the warp-multiple check
# does not apply per dimension.)
ONE_D_BLOCK_KERNELS = {"vector_add", "reduction", "memcpy_bandwidth", "stencil_1d"}


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
            required_block = FIXED_BLOCK_SIZE_KERNELS.get(kernel)
            _require(
                required_block is None or bs == required_block,
                f"kernel '{kernel}' requires block_size={required_block}, got {bs}.",
            )
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


def scenario_warnings(scenarios: list[Scenario]) -> list[str]:
    """Return non-fatal warnings for valid-but-suspicious benchmark configs.

    These do not stop a run; they flag setups that tend to produce noisy or
    misleading numbers (no warmups, too few repeats, non-power-of-two reduction
    blocks, or block sizes that are not multiples of the 32-thread warp).
    """
    warnings: list[str] = []
    for s in scenarios:
        tag = f"{s.kernel} size={s.problem_size} block={s.block_size}"
        if s.warmups == 0:
            warnings.append(f"{tag}: warmups=0 may include one-time JIT/allocation costs in timing.")
        if s.repeats < 5:
            warnings.append(f"{tag}: repeats={s.repeats} is low; statistics may be unstable.")
        if s.kernel in ONE_D_BLOCK_KERNELS and s.block_size % 32 != 0:
            warnings.append(f"{tag}: block_size is not a multiple of the 32-thread warp; expect underutilization.")
        if s.kernel in POWER_OF_TWO_BLOCK_KERNELS and (s.block_size & (s.block_size - 1)) != 0:
            warnings.append(f"{tag}: reduction block_size should be a power of two for the tree reduction.")
    return warnings
