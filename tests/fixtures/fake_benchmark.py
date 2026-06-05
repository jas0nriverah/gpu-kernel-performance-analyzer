"""Deterministic fake benchmark binary for CPU-only testing.

It mimics the JSON contract of the real CUDA harness without a GPU, using per-kernel
byte and FLOP models that match the real kernels' declared work. This lets the analysis,
plotting, and report pipeline be exercised for every kernel (including the zero-FLOP
``memcpy_bandwidth`` case and a tiled-vs-naive GEMM speedup) on CPU-only CI.

Outputs are sample-only and must never be presented as real GPU performance.
"""
from __future__ import annotations

import argparse
import json

_FLOAT_BYTES = 4

# Relative per-kernel timing factors. Tiled GEMM is modeled as faster than naive GEMM so
# the speedup analysis produces a value > 1.0 in fixture runs.
_SPEED_FACTOR = {
    "gemm_tiled": 0.6,
    "gemm_naive": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def byte_and_flop_model(kernel: str, problem_size: int) -> tuple[int, float]:
    """Return (bytes_moved, flops) for a kernel, mirroring the real kernels' work.

    For GEMM kernels ``problem_size`` is the square matrix dimension; for the elementwise
    and streaming kernels it is the element count.
    """
    n = problem_size
    if kernel == "vector_add":
        return 3 * n * _FLOAT_BYTES, float(n)
    if kernel == "memcpy_bandwidth":
        return 2 * n * _FLOAT_BYTES, 0.0
    if kernel == "stencil_1d":
        return 2 * n * _FLOAT_BYTES, 5.0 * n
    if kernel == "reduction":
        return n * _FLOAT_BYTES, float(n)
    if kernel in ("gemm_naive", "gemm_tiled"):
        return 3 * n * n * _FLOAT_BYTES, 2.0 * (n ** 3)
    return max(1, n * 12), float(max(1, n))


def main() -> None:
    args = parse_args()
    factor = _SPEED_FACTOR.get(args.kernel, 1.0)
    base = max(0.01, args.problem_size / 1_000_000.0) * factor
    samples = [base + i * 0.001 for i in range(args.repeats)]
    bytes_moved, flops = byte_and_flop_model(args.kernel, args.problem_size)
    payload = {
        "kernel": args.kernel,
        "problem_size": args.problem_size,
        "block_size": args.block_size,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "verify": args.verify,
        "verification_passed": True,
        "bytes_moved": max(1, bytes_moved),
        "flops": flops,
        "runtime_ms_samples": samples,
        "device": {
            "metadata_available": True,
            "name": "Fake GPU",
            "compute_capability_major": 9,
            "compute_capability_minor": 0,
            "multiprocessor_count": 32,
            "total_global_mem_bytes": 16_000_000_000,
        },
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
