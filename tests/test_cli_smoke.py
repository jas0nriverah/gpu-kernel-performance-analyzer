from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = dict(**__import__("os").environ)
    src_path = _repo_root() / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) if not existing else f"{src_path}{__import__('os').pathsep}{existing}"
    return env


def _write_scenarios(path: Path) -> None:
    payload = {
        "version": 1,
        "sweeps": [
            {
                "kernel": "vector_add",
                "problem_sizes": [1024],
                "block_sizes": [128],
                "warmups": 1,
                "repeats": 3,
                "verify": True,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_sweep_and_validate(tmp_path: Path):
    repo = _repo_root()
    scenario_file = tmp_path / "scenarios.json"
    _write_scenarios(scenario_file)

    fake_binary = repo / "tests" / "fixtures" / "fake_benchmark.py"
    run_dir = tmp_path / "run_artifacts"

    sweep_cmd = [
        sys.executable,
        "-m",
        "gpu_kernel_analyzer",
        "benchmark",
        "sweep",
        "--binary",
        str(fake_binary),
        "--binary-interpreter",
        sys.executable,
        "--scenarios",
        str(scenario_file),
        "--outdir",
        str(run_dir),
    ]
    sweep_proc = subprocess.run(sweep_cmd, cwd=repo, env=_python_env(), capture_output=True, text=True)
    assert sweep_proc.returncode == 0, sweep_proc.stderr

    validate_cmd = [
        sys.executable,
        "-m",
        "gpu_kernel_analyzer",
        "artifacts",
        "validate",
        "--run-dir",
        str(run_dir),
    ]
    validate_proc = subprocess.run(validate_cmd, cwd=repo, env=_python_env(), capture_output=True, text=True)
    assert validate_proc.returncode == 0, validate_proc.stderr

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "timing_samples.csv").exists()
    assert (run_dir / "benchmark_summary.csv").exists()
    assert (run_dir / "metrics_provenance.csv").exists()
