from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--problem-size", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = max(0.01, args.problem_size / 1_000_000.0)
    samples = [base + i * 0.001 for i in range(args.repeats)]
    payload = {
        "kernel": args.kernel,
        "problem_size": args.problem_size,
        "block_size": args.block_size,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "verify": args.verify,
        "verification_passed": True,
        "bytes_moved": max(1, args.problem_size * 12),
        "flops": float(max(1, args.problem_size * 2)),
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
