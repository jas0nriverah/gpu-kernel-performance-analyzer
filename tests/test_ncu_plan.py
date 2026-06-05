from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gpu_kernel_analyzer.ncu import get_metric_queries_for_set


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = _repo_root() / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def test_get_metric_queries_for_default_profiler_set():
    config_path = _repo_root() / "configs" / "ncu_metric_sets.yaml"
    queries = get_metric_queries_for_set("default_profiler_set", config_path)
    assert len(queries) >= 5
    assert all(isinstance(q, str) and q for q in queries)


def test_cli_ncu_plan_prints_commands(tmp_path: Path):
    raw_csv = tmp_path / "ncu_raw.csv"
    norm_csv = tmp_path / "ncu_norm.csv"
    cmd = [
        sys.executable,
        "-m",
        "gpu_kernel_analyzer.cli",
        "profile",
        "ncu-plan",
        "--binary",
        "build/gpu_benchmark",
        "--kernel",
        "vector_add",
        "--problem-size",
        "4194304",
        "--block-size",
        "256",
        "--warmups",
        "10",
        "--repeats",
        "30",
        "--verify",
        "--raw-csv-out",
        str(raw_csv),
        "--normalized-csv",
        str(norm_csv),
        "--run-dir",
        "outputs/demo_real_gpu",
    ]
    proc = subprocess.run(cmd, cwd=_repo_root(), env=_python_env(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "NCU raw capture command:" in proc.stdout
    assert "ncu --target-processes all --csv --page raw --metrics" in proc.stdout
    assert "--kernel vector_add" in proc.stdout
    assert "Normalized import command:" in proc.stdout
    assert "--metric-set default_profiler_set" in proc.stdout
