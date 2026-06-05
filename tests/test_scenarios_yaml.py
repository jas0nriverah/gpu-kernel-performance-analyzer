from pathlib import Path

from gpu_kernel_analyzer.scenarios import load_and_expand_scenarios


def test_yaml_scenarios_expand(tmp_path: Path):
    path = tmp_path / "scenarios.yaml"
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "sweeps:",
                "  - kernel: vector_add",
                "    problem_sizes: [1024, 2048]",
                "    block_sizes: [128]",
                "    warmups: 1",
                "    repeats: 2",
                "    verify: true",
            ]
        ),
        encoding="utf-8",
    )
    expanded = load_and_expand_scenarios(path)
    assert len(expanded) == 2
    assert expanded[0].kernel == "vector_add"


def test_repo_example_yaml_config_loads():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "benchmark_scenarios.yaml"
    expanded = load_and_expand_scenarios(config_path)
    # Lock the expanded scenario count so config edits cannot silently desync the
    # documented validation snapshot (vector_add 2x4 + reduction 2x4 + gemm_naive 2 +
    # gemm_tiled 2 + memcpy_bandwidth 2 + stencil_1d 2 = 24).
    assert len(expanded) == 24
    kernels = {s.kernel for s in expanded}
    assert kernels == {
        "vector_add",
        "reduction",
        "gemm_naive",
        "gemm_tiled",
        "memcpy_bandwidth",
        "stencil_1d",
    }


def test_repo_smoke_config_loads():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "smoke_new_kernels.yaml"
    expanded = load_and_expand_scenarios(config_path)
    assert {s.kernel for s in expanded} == {"memcpy_bandwidth", "stencil_1d"}
