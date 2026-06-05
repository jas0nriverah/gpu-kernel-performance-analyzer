from __future__ import annotations

from pathlib import Path

import pytest

from gpu_kernel_analyzer.scenarios import (
    SUPPORTED_KERNELS,
    Scenario,
    load_and_expand_scenarios,
    scenario_warnings,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_new_kernels_are_supported():
    assert {"memcpy_bandwidth", "stencil_1d"}.issubset(SUPPORTED_KERNELS)


def test_gemm_tiled_requires_block_size_16(tmp_path: Path):
    path = _write(
        tmp_path / "bad.yaml",
        "\n".join(
            [
                "version: 1",
                "sweeps:",
                "  - kernel: gemm_tiled",
                "    problem_sizes: [256]",
                "    block_sizes: [32]",
                "    warmups: 5",
                "    repeats: 15",
            ]
        ),
    )
    with pytest.raises(ValueError, match="block_size=16"):
        load_and_expand_scenarios(path)


def test_warning_for_zero_warmups_and_low_repeats():
    scenarios = [Scenario("vector_add", 1024, 256, warmups=0, repeats=2, verify=False)]
    warnings = scenario_warnings(scenarios)
    assert any("warmups=0" in w for w in warnings)
    assert any("repeats=2" in w for w in warnings)


def test_warning_for_non_warp_multiple_block_on_1d_kernel():
    scenarios = [Scenario("vector_add", 1024, 100, warmups=10, repeats=30, verify=False)]
    warnings = scenario_warnings(scenarios)
    assert any("warp" in w for w in warnings)


def test_no_warp_warning_for_gemm_2d_block():
    scenarios = [Scenario("gemm_tiled", 256, 16, warmups=10, repeats=30, verify=False)]
    warnings = scenario_warnings(scenarios)
    assert not any("warp" in w for w in warnings)


def test_clean_config_has_no_warnings():
    scenarios = [Scenario("vector_add", 1048576, 256, warmups=10, repeats=30, verify=True)]
    assert scenario_warnings(scenarios) == []
