from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gpu_kernel_analyzer.runner import run_binary_for_scenario
from gpu_kernel_analyzer.scenarios import Scenario


def _scenario() -> Scenario:
    return Scenario(
        kernel="vector_add",
        problem_size=1024,
        block_size=128,
        warmups=1,
        repeats=2,
        verify=False,
    )


def test_missing_binary_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        run_binary_for_scenario(binary=missing, scenario=_scenario())


def test_nonzero_exit_raises(tmp_path: Path):
    script = tmp_path / "boom.py"
    script.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        run_binary_for_scenario(binary=script, scenario=_scenario(), interpreter=sys.executable)


def test_empty_output_raises(tmp_path: Path):
    script = tmp_path / "silent.py"
    script.write_text("pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        run_binary_for_scenario(binary=script, scenario=_scenario(), interpreter=sys.executable)


def test_invalid_json_raises(tmp_path: Path):
    script = tmp_path / "bad.py"
    script.write_text("print('not json at all')\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        run_binary_for_scenario(binary=script, scenario=_scenario(), interpreter=sys.executable)


def test_non_object_json_raises(tmp_path: Path):
    script = tmp_path / "array.py"
    script.write_text("print('[1, 2, 3]')\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        run_binary_for_scenario(binary=script, scenario=_scenario(), interpreter=sys.executable)


def test_timeout_terminates_child(tmp_path: Path):
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="timed out"):
        run_binary_for_scenario(script, _scenario(), sys.executable, timeout_seconds=0.1)


@pytest.mark.parametrize("change, message", [
    ({"kernel": "reduction"}, "requested kernel"),
    ({"verification_passed": False}, "verification failed"),
    ({"verification_passed": "true"}, "verification failed"),
    ({"verification_passed": None}, "verification failed"),
    ({"runtime_ms_samples": [1.0]}, "sample count"),
    ({"runtime_ms_samples": [1.0, float("nan")]}, "finite positive"),
    ({"runtime_ms_samples": [1.0, 0.0]}, "finite positive"),
])
def test_rejects_invalid_result_contract(tmp_path: Path, change: dict, message: str):
    scenario = Scenario("vector_add", 1024, 128, 1, 2, True)
    payload = dict(vars(scenario), verification_passed=True, bytes_moved=12288, flops=1024, runtime_ms_samples=[1.0, 2.0])
    payload.update(change)
    script = tmp_path / "invalid_result.py"
    script.write_text(f"print({json.dumps(payload)!r})\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        run_binary_for_scenario(script, scenario, sys.executable)


def test_sweep_refuses_to_overwrite_evidence(tmp_path: Path):
    import subprocess

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    evidence = run_dir / "run_manifest.json"
    evidence.write_text('{"important": "existing evidence"}')
    before = evidence.read_bytes()
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_kernel_analyzer", "benchmark", "sweep",
         "--binary", "tests/fixtures/fake_benchmark.py", "--binary-interpreter", sys.executable,
         "--scenarios", "configs/benchmark_scenarios.yaml", "--outdir", str(run_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "not empty" in proc.stderr
    assert evidence.read_bytes() == before
    assert list(run_dir.iterdir()) == [evidence]
