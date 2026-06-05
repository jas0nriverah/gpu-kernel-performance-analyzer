from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = _repo_root() / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _write_scenarios(path: Path) -> None:
    payload = {
        "version": 1,
        "sweeps": [
            {
                "kernel": "gemm_naive",
                "problem_sizes": [256],
                "block_sizes": [16],
                "warmups": 1,
                "repeats": 3,
                "verify": True,
            },
            {
                "kernel": "gemm_tiled",
                "problem_sizes": [256],
                "block_sizes": [16],
                "warmups": 1,
                "repeats": 3,
                "verify": True,
            },
            {
                "kernel": "memcpy_bandwidth",
                "problem_sizes": [1048576],
                "block_sizes": [256],
                "warmups": 1,
                "repeats": 3,
                "verify": True,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=_python_env(), capture_output=True, text=True)


def test_analyze_full_generates_report_and_plots(tmp_path: Path):
    repo = _repo_root()
    scenario_file = tmp_path / "scenarios.json"
    _write_scenarios(scenario_file)
    fake_binary = repo / "tests" / "fixtures" / "fake_benchmark.py"
    run_dir = tmp_path / "run"

    sweep = _run(
        [
            sys.executable, "-m", "gpu_kernel_analyzer", "benchmark", "sweep",
            "--binary", str(fake_binary), "--binary-interpreter", sys.executable,
            "--scenarios", str(scenario_file), "--outdir", str(run_dir),
        ],
        repo,
    )
    assert sweep.returncode == 0, sweep.stderr

    analyze = _run(
        [
            sys.executable, "-m", "gpu_kernel_analyzer", "analyze", "full",
            "--run-dir", str(run_dir),
        ],
        repo,
    )
    assert analyze.returncode == 0, analyze.stderr

    assert (run_dir / "REPORT.md").exists()
    assert (run_dir / "analysis_heuristics.csv").exists()
    assert (run_dir / "analysis_speedup.csv").exists()
    assert (run_dir / "plots" / "roofline.png").exists()
    assert (run_dir / "plots" / "effective_bandwidth_vs_size_memory_kernels.png").exists()

    # Tiled GEMM should be reported faster than naive GEMM in the fixture model.
    with (run_dir / "analysis_speedup.csv").open(encoding="utf-8", newline="") as fh:
        speedups = list(csv.DictReader(fh))
    assert speedups, "expected at least one optimized-vs-baseline pair"
    assert any(float(r["speedup_runtime"]) > 1.0 for r in speedups)

    report_text = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "gemm_tiled" in report_text
