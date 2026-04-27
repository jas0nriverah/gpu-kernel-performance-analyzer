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
    assert len(expanded) > 0
