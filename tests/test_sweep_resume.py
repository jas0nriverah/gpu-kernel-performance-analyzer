from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gpu_kernel_analyzer import cli
from gpu_kernel_analyzer.resumability import CHECKPOINT_NAME, current_device_identity, digest, seal


def _scenario_file(path: Path, sizes=(64, 128)) -> Path:
    path.write_text(json.dumps({"sweeps": [{"kernel": "vector_add", "problem_sizes": list(sizes), "block_sizes": [32], "warmups": 1, "repeats": 2, "verify": True}]}))
    return path


def _args(binary: Path, scenario: Path, outdir: Path, resume=False):
    return argparse.Namespace(binary=str(binary), scenarios=str(scenario), outdir=str(outdir), run_id="resume-test", dry_run=False, resume=resume, binary_interpreter=None, timeout_seconds=5.0)


def test_interrupted_sweep_resumes_completed_scenarios(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_text("fixed binary")
    scenario = _scenario_file(tmp_path / "scenarios.json")
    outdir = tmp_path / "run"
    calls = []

    class Result:
        def __init__(self, size):
            self.raw = {"kernel": "vector_add", "problem_size": size, "block_size": 32, "warmups": 1, "repeats": 2, "verify": True, "verification_passed": True, "bytes_moved": 768, "flops": 64, "runtime_ms_samples": [1.0, 1.1], "device": {"metadata_available": True, "name": "Fake GPU", "uuid": "fixture-0"}}

    def runner(binary, scenario, **kwargs):
        calls.append(scenario.problem_size)
        if scenario.problem_size == 128 and calls.count(128) == 1:
            raise RuntimeError("simulated interruption")
        return Result(scenario.problem_size)

    monkeypatch.setattr(cli, "run_binary_for_scenario", runner)
    device = {"metadata_available": True, "name": "Fake GPU", "uuid": "fixture-0"}
    monkeypatch.setattr(cli, "probe_live_device_identity", lambda *a: current_device_identity(device))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cli.run_sweep(_args(binary, scenario, outdir))
    assert not outdir.exists()
    partial = tmp_path / ".run.partial"
    assert (partial / CHECKPOINT_NAME).exists()
    assert cli.run_sweep(_args(binary, scenario, outdir, resume=True)) == 0
    assert calls == [64, 128, 128]
    assert (outdir / "benchmark_summary.csv").exists()
    assert not partial.exists()


def test_resume_rejects_checkpoint_tampering(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_text("fixed binary")
    scenario = _scenario_file(tmp_path / "scenarios.json", sizes=(64,))
    outdir = tmp_path / "run"

    class Result:
        raw = {"kernel": "vector_add", "problem_size": 64, "block_size": 32, "warmups": 1, "repeats": 2, "verify": True, "verification_passed": True, "bytes_moved": 768, "flops": 64, "runtime_ms_samples": [1.0, 1.1], "device": {"metadata_available": True, "name": "Fake GPU"}}

    def runner(*args, **kwargs):
        return Result()

    monkeypatch.setattr(cli, "run_binary_for_scenario", runner)
    monkeypatch.setattr(cli, "probe_live_device_identity", lambda *a: current_device_identity(Result.raw["device"]))
    # Make a checkpoint by interrupting after the scenario has been committed during artifact construction.
    original_write_csv = cli.write_csv
    monkeypatch.setattr(cli, "write_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("publish fail")))
    with pytest.raises(RuntimeError, match="publish fail"):
        cli.run_sweep(_args(binary, scenario, outdir))
    path = tmp_path / ".run.partial" / CHECKPOINT_NAME
    payload = json.loads(path.read_text())
    payload["completed"][0]["raw"]["runtime_ms_samples"][0] = 7.0
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(cli, "write_csv", original_write_csv)
    with pytest.raises(ValueError, match="integrity"):
        cli.run_sweep(_args(binary, scenario, outdir, resume=True))


def test_resume_rejects_live_device_change(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_text("fixed binary")
    scenario = _scenario_file(tmp_path / "scenarios.json", sizes=(64,))
    outdir = tmp_path / "run"
    device = {"metadata_available": True, "name": "Fake GPU", "uuid": "fixture-a"}

    class Result:
        raw = {"kernel": "vector_add", "problem_size": 64, "block_size": 32, "warmups": 1, "repeats": 2, "verify": True, "verification_passed": True, "bytes_moved": 768, "flops": 64, "runtime_ms_samples": [1.0, 1.1], "device": device}

    monkeypatch.setattr(cli, "run_binary_for_scenario", lambda *a, **k: Result())
    original_write_csv = cli.write_csv
    monkeypatch.setattr(cli, "write_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("publish fail")))
    with pytest.raises(RuntimeError, match="publish fail"):
        cli.run_sweep(_args(binary, scenario, outdir))
    monkeypatch.setattr(cli, "write_csv", original_write_csv)
    monkeypatch.setattr(cli, "probe_live_device_identity", lambda *a: "cuda-uuid:other-device")
    with pytest.raises(ValueError, match="GPU identity differs"):
        cli.run_sweep(_args(binary, scenario, outdir, resume=True))


def test_resume_rejects_changed_config_and_run_id(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_text("fixed binary")
    scenario = _scenario_file(tmp_path / "scenarios.json", sizes=(64,))
    outdir = tmp_path / "run"

    class Result:
        raw = {"kernel": "vector_add", "problem_size": 64, "block_size": 32, "warmups": 1, "repeats": 2, "verify": True, "verification_passed": True, "bytes_moved": 768, "flops": 64, "runtime_ms_samples": [1.0, 1.1], "device": {"metadata_available": True, "name": "Fake GPU", "uuid": "fixture-a"}}

    monkeypatch.setattr(cli, "run_binary_for_scenario", lambda *a, **k: Result())
    original_write_csv = cli.write_csv
    monkeypatch.setattr(cli, "write_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("publish fail")))
    with pytest.raises(RuntimeError):
        cli.run_sweep(_args(binary, scenario, outdir))
    monkeypatch.setattr(cli, "write_csv", original_write_csv)
    changed = _scenario_file(tmp_path / "changed.json", sizes=(128,))
    with pytest.raises(ValueError, match="incompatible"):
        cli.run_sweep(_args(binary, changed, outdir, resume=True))
    bad_run_id = _args(binary, scenario, outdir, resume=True)
    bad_run_id.run_id = "different"
    with pytest.raises(ValueError, match="run-id"):
        cli.run_sweep(bad_run_id)


def test_resume_rejects_binary_change(tmp_path, monkeypatch):
    binary = tmp_path / "binary"
    binary.write_text("version one")
    scenario = _scenario_file(tmp_path / "scenarios.json", sizes=(64,))
    outdir = tmp_path / "run"

    class Result:
        raw = {"kernel": "vector_add", "problem_size": 64, "block_size": 32, "warmups": 1, "repeats": 2, "verify": True, "verification_passed": True, "bytes_moved": 768, "flops": 64, "runtime_ms_samples": [1.0, 1.1], "device": {"metadata_available": True, "name": "Fake GPU", "uuid": "fixture-a"}}

    monkeypatch.setattr(cli, "run_binary_for_scenario", lambda *a, **k: Result())
    original_write_csv = cli.write_csv
    monkeypatch.setattr(cli, "write_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("publish fail")))
    with pytest.raises(RuntimeError):
        cli.run_sweep(_args(binary, scenario, outdir))
    monkeypatch.setattr(cli, "write_csv", original_write_csv)
    binary.write_text("version two")
    with pytest.raises(ValueError, match="incompatible"):
        cli.run_sweep(_args(binary, scenario, outdir, resume=True))


def test_sweep_lock_releases_after_process_exit(tmp_path):
    import multiprocessing

    from gpu_kernel_analyzer.resumability import sweep_lock

    run_dir = tmp_path / "run"
    ready = multiprocessing.Event()
    release = multiprocessing.Event()

    def hold_lock():
        with sweep_lock(run_dir):
            ready.set()
            release.wait(10)

    process = multiprocessing.Process(target=hold_lock)
    process.start()
    assert ready.wait(5)
    with pytest.raises(RuntimeError, match="Another sweep"):
        with sweep_lock(run_dir):
            pass
    process.kill()
    process.join(5)
    assert process.exitcode is not None
    with sweep_lock(run_dir):
        pass


def _leave_checkpoint_after_publish_failure(tmp_path, monkeypatch, scenario, outdir):
    binary = tmp_path / "binary"
    binary.write_text("fixed binary")
    device = {"metadata_available": True, "name": "Fake GPU", "uuid": "fixture-a"}

    class Result:
        raw = {"kernel": "vector_add", "problem_size": 64, "block_size": 32, "warmups": 1, "repeats": 2, "verify": True, "verification_passed": True, "bytes_moved": 768, "flops": 64, "runtime_ms_samples": [1.0, 1.1], "device": device}

    calls = []
    monkeypatch.setattr(cli, "run_binary_for_scenario", lambda *a, **k: (calls.append(1) or Result()))
    monkeypatch.setattr(cli, "probe_live_device_identity", lambda *a: current_device_identity(device))
    original_write_csv = cli.write_csv
    monkeypatch.setattr(cli, "write_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("publish fail")))
    with pytest.raises(RuntimeError, match="publish fail"):
        cli.run_sweep(_args(binary, scenario, outdir))
    monkeypatch.setattr(cli, "write_csv", original_write_csv)
    return binary, calls


def test_resume_revalidates_resealed_checkpoint_scenario_and_timing(tmp_path, monkeypatch):
    scenario = _scenario_file(tmp_path / "scenarios.json", sizes=(64,))
    outdir = tmp_path / "run"
    _binary, _calls = _leave_checkpoint_after_publish_failure(tmp_path, monkeypatch, scenario, outdir)
    path = tmp_path / ".run.partial" / CHECKPOINT_NAME
    original = json.loads(path.read_text())

    for mutate, message in [
        (lambda raw: raw.update(kernel="reduction"), "does not match requested kernel"),
        (lambda raw: raw.update(runtime_ms_samples=[1.0]), "sample count"),
    ]:
        payload = json.loads(json.dumps(original))
        raw = payload["completed"][0]["raw"]
        mutate(raw)
        payload["completed"][0]["raw_sha256"] = digest(raw)
        path.write_text(json.dumps(seal(payload)))
        with pytest.raises(RuntimeError, match=message):
            cli.run_sweep(_args(_binary, scenario, outdir, resume=True))


def test_resume_after_publication_failure_does_not_rerun_completed_scenario(tmp_path, monkeypatch):
    scenario = _scenario_file(tmp_path / "scenarios.json", sizes=(64,))
    outdir = tmp_path / "run"
    binary, calls = _leave_checkpoint_after_publish_failure(tmp_path, monkeypatch, scenario, outdir)

    assert calls == [1]
    assert cli.run_sweep(_args(binary, scenario, outdir, resume=True)) == 0
    assert calls == [1]
    assert (outdir / "benchmark_summary.csv").exists()
