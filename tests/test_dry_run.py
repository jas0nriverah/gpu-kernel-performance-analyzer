from __future__ import annotations

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


def test_dry_run_expands_without_running_binary(tmp_path: Path):
    repo = _repo_root()
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(
        json.dumps(
            {
                "version": 1,
                "sweeps": [
                    {
                        "kernel": "vector_add",
                        "problem_sizes": [1024, 2048],
                        "block_sizes": [128, 256],
                        "warmups": 1,
                        "repeats": 2,
                        "verify": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    proc = subprocess.run(
        [
            sys.executable, "-m", "gpu_kernel_analyzer", "benchmark", "sweep",
            "--binary", "/nonexistent/binary",
            "--scenarios", str(scenario_file),
            "--outdir", str(run_dir),
            "--dry-run",
        ],
        cwd=repo,
        env=_python_env(),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Scenarios expanded: 4" in proc.stdout
    # Dry run must not invoke the (nonexistent) binary or write artifacts.
    assert not (run_dir / "benchmark_summary.csv").exists()
