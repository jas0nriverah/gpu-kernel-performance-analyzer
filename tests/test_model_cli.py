from __future__ import annotations

import csv
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


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gpu_kernel_analyzer", *args],
        cwd=_repo_root(),
        env=_python_env(),
        capture_output=True,
        text=True,
    )


_SUMMARY_HEADER = (
    "run_id,kernel,problem_size,block_size,warmups,repeats,verification_passed,"
    "runtime_ms_mean,runtime_ms_median,runtime_ms_min,runtime_ms_max,runtime_ms_p95,"
    "runtime_ms_stddev,runtime_ms_cv,effective_bandwidth_GBps,effective_GFLOPs,arithmetic_intensity"
)


def _seed_summary(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [_SUMMARY_HEADER]
    for problem_size in (1048576, 4194304):
        for block_size, runtime in ((64, 0.40), (128, 0.30), (256, 0.22), (512, 0.18)):
            scale = problem_size / 1048576
            rt = runtime * scale
            bw = 900.0 / scale
            lines.append(
                f"r1,vector_add,{problem_size},{block_size},10,30,True,{rt},{rt},{rt},{rt},{rt},0.0,0.02,{bw},0.0,0.0"
            )
    (run_dir / "benchmark_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_model_train_predict_recommend_and_advise(tmp_path: Path):
    run_dir = tmp_path / "run"
    _seed_summary(run_dir)
    model_path = tmp_path / "perf_model.json"

    train = _run(["model", "train", "--run-dir", str(run_dir), "--model-out", str(model_path)])
    assert train.returncode == 0, train.stderr
    assert model_path.exists()

    # Without --predictions-out the entire stdout is the JSON prediction object.
    predict = _run(
        [
            "model", "predict", "--model", str(model_path),
            "--kernel", "vector_add", "--problem-size", "1048576", "--block-size", "256",
        ]
    )
    assert predict.returncode == 0, predict.stderr
    payload = json.loads(predict.stdout)
    assert payload["status"] == "predicted"
    assert payload["predicted"]["runtime_ms_mean"] > 0.0

    # With --predictions-out a CSV artifact of predicted rows is written.
    predictions_csv = tmp_path / "model_predictions.csv"
    predict_csv = _run(
        [
            "model", "predict", "--model", str(model_path),
            "--kernel", "vector_add", "--problem-size", "1048576", "--block-size", "256",
            "--predictions-out", str(predictions_csv),
        ]
    )
    assert predict_csv.returncode == 0, predict_csv.stderr
    with predictions_csv.open(encoding="utf-8", newline="") as fh:
        pred_rows = list(csv.DictReader(fh))
    assert pred_rows and all(r["status"] == "predicted" for r in pred_rows)

    recommend = _run(
        [
            "model", "recommend-block-size", "--model", str(model_path),
            "--kernel", "vector_add", "--problem-size", "4194304", "--candidates", "64,128,256,512",
        ]
    )
    assert recommend.returncode == 0, recommend.stderr
    assert "Recommended block_size" in recommend.stdout

    advise = _run(["advise", "--run-dir", str(run_dir)])
    assert advise.returncode == 0, advise.stderr
    assert "[autotune]" in advise.stdout
    assert (run_dir / "advisor_report.md").exists()
