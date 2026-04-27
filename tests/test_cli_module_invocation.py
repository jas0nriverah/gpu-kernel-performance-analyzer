from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = _repo_root() / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def test_module_invocation_help_outputs_usage():
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_kernel_analyzer.cli", "--help"],
        cwd=_repo_root(),
        env=_python_env(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "GPU Kernel Performance Analyzer CLI" in proc.stdout
    assert "benchmark" in proc.stdout


def test_module_invocation_benchmark_help_outputs_usage():
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_kernel_analyzer.cli", "benchmark", "--help"],
        cwd=_repo_root(),
        env=_python_env(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout
    assert "benchmark" in proc.stdout
    assert "sweep" in proc.stdout


def test_module_invocation_ncu_detect_outputs_json():
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_kernel_analyzer.cli", "profile", "ncu-detect"],
        cwd=_repo_root(),
        env=_python_env(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert "available" in payload
    assert "status" in payload
