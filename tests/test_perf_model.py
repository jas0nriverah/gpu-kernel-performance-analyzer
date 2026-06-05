from __future__ import annotations

import json
from pathlib import Path

from gpu_kernel_analyzer.artifacts import SUMMARY_COLUMNS, write_csv
from gpu_kernel_analyzer.metrics import STATUS_PREDICTED
from gpu_kernel_analyzer.perf_model import (
    PerfModel,
    cross_validate,
    load_training_rows,
    prediction_rows,
    recommend_block_size,
    train_perf_model,
)


def _summary_row(kernel: str, problem_size: int, block_size: int, runtime: float, bandwidth: float) -> dict:
    row = {col: "" for col in SUMMARY_COLUMNS}
    row.update(
        {
            "run_id": "r1",
            "kernel": kernel,
            "problem_size": problem_size,
            "block_size": block_size,
            "warmups": 10,
            "repeats": 30,
            "verification_passed": True,
            "runtime_ms_mean": runtime,
            "runtime_ms_median": runtime,
            "runtime_ms_min": runtime,
            "runtime_ms_max": runtime,
            "runtime_ms_p95": runtime,
            "runtime_ms_stddev": 0.0,
            "runtime_ms_cv": 0.0,
            "effective_bandwidth_GBps": bandwidth,
            "effective_GFLOPs": 0.0,
            "arithmetic_intensity": 0.0,
        }
    )
    return row


def _seed_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # vector_add: smaller block_size is modeled as slower so 512 should win.
    for problem_size in (1_048_576, 4_194_304):
        for block_size, runtime in ((64, 0.40), (128, 0.30), (256, 0.22), (512, 0.18)):
            scale = problem_size / 1_048_576
            rows.append(
                _summary_row("vector_add", problem_size, block_size, runtime * scale, 900.0 / scale)
            )
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)


def test_train_and_predict_output_format(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_run_dir(run_dir)
    rows = load_training_rows([run_dir])
    model = train_perf_model(rows)

    assert model.kernel_vocab == ["vector_add"]
    assert "runtime_ms_mean" in model.targets
    assert "effective_bandwidth_GBps" in model.targets

    predicted = model.predict("vector_add", 1_048_576, 256)
    assert set(predicted.keys()) == {"runtime_ms_mean", "effective_bandwidth_GBps"}
    assert predicted["runtime_ms_mean"] > 0.0
    assert predicted["effective_bandwidth_GBps"] > 0.0


def test_recommend_block_size_prefers_faster_block(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_run_dir(run_dir)
    model = train_perf_model(load_training_rows([run_dir]))
    rec = recommend_block_size(model, "vector_add", 1_048_576, [64, 128, 256, 512])
    # The seeded data makes larger blocks faster, so 512 should be recommended.
    assert rec.recommended_block_size == 512
    assert rec.data_backed is True
    assert len(rec.candidates) == 4


def test_recommend_warns_when_block_not_varied(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _summary_row("gemm_tiled", 256, 16, 0.05, 40.0),
        _summary_row("gemm_tiled", 512, 16, 0.09, 45.0),
    ]
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)
    model = train_perf_model(load_training_rows([run_dir]))
    rec = recommend_block_size(model, "gemm_tiled", 512, [16, 32])
    assert rec.data_backed is False


def test_cross_validate_small_data_warning(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _summary_row("vector_add", 1_048_576, 128, 0.30, 900.0),
        _summary_row("vector_add", 1_048_576, 256, 0.22, 1200.0),
        _summary_row("vector_add", 4_194_304, 128, 1.20, 900.0),
    ]
    write_csv(run_dir / "benchmark_summary.csv", rows=rows, fieldnames=SUMMARY_COLUMNS)
    cv = cross_validate(load_training_rows([run_dir]))
    assert cv["enough_data"] is False
    assert cv["n_rows"] == 3


def test_model_round_trip(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_run_dir(run_dir)
    model = train_perf_model(load_training_rows([run_dir]))
    model_path = tmp_path / "perf_model.json"
    model_path.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    reloaded = PerfModel.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    assert reloaded.kernel_vocab == model.kernel_vocab
    a = model.predict("vector_add", 1_048_576, 256)
    b = reloaded.predict("vector_add", 1_048_576, 256)
    assert abs(a["runtime_ms_mean"] - b["runtime_ms_mean"]) < 1e-9


def test_prediction_rows_use_predicted_status():
    rows = prediction_rows(
        "vector_add", 1_048_576, 256, {"runtime_ms_mean": 0.2, "effective_bandwidth_GBps": 1000.0}, "perf_model:numpy"
    )
    assert rows
    assert all(r["status"] == STATUS_PREDICTED for r in rows)
    assert all(r["source"] == "perf_model:numpy" for r in rows)
