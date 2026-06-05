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
