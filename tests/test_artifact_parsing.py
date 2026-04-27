from pathlib import Path

from gpu_kernel_analyzer.artifacts import read_csv_rows, write_csv


def test_artifact_csv_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    rows = [
        {"run_id": "r1", "kernel": "vector_add", "runtime_ms": 0.12},
        {"run_id": "r1", "kernel": "reduction", "runtime_ms": 0.20},
    ]
    write_csv(csv_path, rows=rows, fieldnames=["run_id", "kernel", "runtime_ms"])
    loaded = read_csv_rows(csv_path)
    assert len(loaded) == 2
    assert loaded[0]["kernel"] == "vector_add"
    assert loaded[1]["runtime_ms"] == "0.2"
