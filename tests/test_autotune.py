import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from gpu_kernel_analyzer import autotune
from gpu_kernel_analyzer.artifacts import sha256_file


def candidate(cid, kernel, size, runtime, energy):
    return {"candidate_id": cid, "kernel": kernel, "problem_size": size,
            "workload": autotune.workload_key(kernel, size), "valid": True,
            "runtime_s_per_launch_median": runtime, "energy_j_per_launch_median": energy}


def test_selection_limits_speed_tradeoff_and_separates_workloads():
    rows = [candidate("fast", "gemm_naive", 64, 1.0, 10),
            candidate("efficient", "gemm_tiled", 64, 1.04, 7),
            candidate("too_slow", "gemm_tiled", 64, 1.2, 2),
            candidate("other_size", "gemm_tiled", 128, 1.2, 2),
            candidate("vector", "vector_add", 64, 0.1, 1)]
    chosen = autotune.choose_candidates(rows, 0.05)
    assert chosen["gemm:64"]["selected_candidate_id"] == "efficient"
    assert set(chosen) == {"gemm:64", "gemm:128", "vector_add:64"}
    rows[0]["workload"] = "vector_add:64"
    with pytest.raises(ValueError, match="incompatible"):
        autotune.choose_candidates(rows)


def test_equal_energy_tie_selects_fast_candidate_and_confirmable_frontier():
    rows = [candidate("slow", "vector_add", 64, 1.03, 7.0),
            candidate("fast", "vector_add", 64, 1.0, 7.0)]
    selection = autotune.choose_candidates(rows, .05)["vector_add:64"]
    assert selection["selected_candidate_id"] == "fast"
    assert "fast" in selection["pareto_frontier"]


def test_confirmed_energy_dispersion_tie_does_not_claim_unique_winner():
    measured = [candidate("fast", "vector_add", 64, .288, .13758),
                candidate("slower", "vector_add", 64, .296, .137728)]
    measured[0]["energy_j_per_launch_mad"] = .00058
    measured[1]["energy_j_per_launch_mad"] = .00006
    confirmed = [candidate("fast", "vector_add", 64, .28835, .136985),
                 candidate("slower", "vector_add", 64, .2962, .136970)]
    confirmed[0].update(confirmation_passed=True, energy_j_per_launch_mad=.00005)
    confirmed[1].update(confirmation_passed=True, energy_j_per_launch_mad=.00004)
    result = autotune.compare_confirmed_energy(measured, confirmed, ["fast", "slower"], "fast")
    assert result["indistinguishable_energy_candidates"] == ["fast", "slower"]
    assert result["fastest_practical_recommendation_if_tied"] == "fast"
    assert result["unique_energy_minimum_confirmed"] is False


def test_discovery_overlap_still_prevents_unique_claim_after_confirmation():
    measured = [candidate("fast", "vector_add", 64, .288, .13758),
                candidate("slower", "vector_add", 64, .296, .137728)]
    measured[0]["energy_j_per_launch_mad"] = .00058
    measured[1]["energy_j_per_launch_mad"] = .00006
    confirmed = [candidate("fast", "vector_add", 64, .28835, .136985),
                 candidate("slower", "vector_add", 64, .2962, .138219)]
    confirmed[0].update(confirmation_passed=True, energy_j_per_launch_mad=.000052)
    confirmed[1].update(confirmation_passed=True, energy_j_per_launch_mad=.000217)
    result = autotune.compare_confirmed_energy(measured, confirmed, ["fast", "slower"], "fast")
    assert result["indistinguishable_energy_candidates"] == ["fast", "slower"]
    assert result["unique_energy_minimum_confirmed"] is False


def test_confirmed_energy_ranking_reversal_is_not_validated():
    measured = [candidate("selected", "vector_add", 64, .1, .1),
                candidate("other", "vector_add", 64, .102, .2)]
    for row in measured:
        row["energy_j_per_launch_mad"] = 0.0
    confirmed = [candidate("selected", "vector_add", 64, .1, .2),
                 candidate("other", "vector_add", 64, .102, .1)]
    for row in confirmed:
        row.update(confirmation_passed=True, energy_j_per_launch_mad=0.0)
    result = autotune.compare_confirmed_energy(measured, confirmed, ["selected", "other"], "selected")
    assert result["confirmed_lowest_energy_candidate_id"] == "other"
    assert result["unique_energy_minimum_confirmed"] is False


def test_missing_confirmation_for_eligible_competitor_cannot_make_unique_winner():
    measured = [candidate("selected", "vector_add", 64, .1, .1),
                candidate("competitor", "vector_add", 64, .104, .2)]
    confirmed = [candidate("selected", "vector_add", 64, .1, .1)]
    confirmed[0].update(confirmation_passed=True, energy_j_per_launch_mad=0.)
    for row in measured:
        row["energy_j_per_launch_mad"] = 0.
    result = autotune.compare_confirmed_energy(measured, confirmed,
                                               ["selected", "competitor"], "selected")
    assert result["unique_energy_minimum_confirmed"] is False


def test_capture_rejects_tampered_and_incomplete_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    root.mkdir()
    summary = [{"kernel": "vector_add", "problem_size": 16, "block_size": 128,
                "trial": 1, "verification_passed": True,
                "estimated_j_per_launch": 1.0, "sustained_duration_s": 4.0,
                "launches_per_second": 2.0, "mean_power_w": 2.0,
                "window": {"launches": 8}}]
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "workers").mkdir()
    worker = {"device": {"uuid": "GPU-test"}, "kernel": "vector_add", "problem_size": 16,
              "block_size": 128, "warmups": 1, "repeats": 2, "verify": True,
              "verification_passed": True, "bytes_moved": 10, "flops": 0,
              "runtime_ms_samples": [1.0, 1.1]}
    (root / "workers" / "trial-1-worker.json").write_text(json.dumps(worker))
    for filename in ("summary.csv", "telemetry.csv", "steady_telemetry.csv"):
        (root / filename).write_text("evidence\n")
    hashes = {name: sha256_file(root / name) for name in
              ("summary.csv", "telemetry.csv", "steady_telemetry.csv", "workers/trial-1-worker.json")}
    hashes["summary.json"] = "bad"
    manifest = {"status": "complete", "backend": "cuda", "power_scope": "gpu_board",
                "binary_sha256": "b", "device": {"gpu_uuid": "GPU-test"},
                "protocol": {"trials": 1, "seed": 1, "duration_seconds": 5.,
                             "interval_seconds": .2, "idle_seconds": 3.,
                             "trim_seconds": 1., "timeout_seconds": 10.},
                "artifacts_sha256": hashes}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tampered"):
        autotune.validate_capture(root, {"kernel": "vector_add", "problem_size": 16,
                                         "block_size": 128, "trials": 1,
                                         "binary_sha256": "b", "gpu_uuid": "GPU-test",
                                         "warmups": 1, "repeats": 2,
                                         "protocol": manifest["protocol"]})


@pytest.mark.parametrize("expected, message", [
    ({"binary_sha256": "other"}, "binary identity"),
    ({"gpu_uuid": "GPU-other"}, "GPU identity"),
    ({"protocol": {"seed": 9}}, "protocol"),
])
def test_capture_binds_device_binary_and_protocol(tmp_path, expected, message):
    root = tmp_path / "capture"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"status": "complete", "backend": "cuda",
        "power_scope": "gpu_board", "binary_sha256": "b", "device": {"gpu_uuid": "GPU-test"},
        "protocol": {"seed": 1}}))
    with pytest.raises((KeyError, ValueError), match=message):
        autotune.validate_capture(root, {"kernel": "vector_add", "problem_size": 16,
            "block_size": 128, "trials": 1, "binary_sha256": "b", "gpu_uuid": "GPU-test",
            "warmups": 1, "repeats": 2, "protocol": {"seed": 1}, **expected})


def test_capture_rejects_missing_trial_coverage_from_summary(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    summary = root / "summary.json"
    summary.write_text("[]")
    manifest = {"status": "complete", "backend": "cuda", "power_scope": "gpu_board",
                "binary_sha256": "b", "device": {"gpu_uuid": "GPU-test"},
                "protocol": {"trials": 2, "seed": 1},
                "artifacts_sha256": {"summary.json": sha256_file(summary)}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="artifact hash coverage"):
        autotune.validate_capture(root, {"kernel": "vector_add", "problem_size": 16,
            "block_size": 128, "trials": 2, "binary_sha256": "b", "gpu_uuid": "GPU-test",
            "warmups": 1, "repeats": 2, "protocol": {"trials": 2, "seed": 1}})


def test_workflow_preflight_rejection_and_independent_confirmation(tmp_path, monkeypatch):
    binary = tmp_path / "bench"
    binary.write_text("fixture")
    args = Namespace(binary=str(binary), kernels=["vector_add"], problem_sizes=[16, 32],
        block_sizes=[128, 256, 512], outdir=str(tmp_path / "out"), device="GPU-test",
        trials=2, confirm_trials=2, speed_tolerance=.05, seed=10, warmups=1, repeats=2,
        timeout_seconds=5, duration_seconds=5., interval_seconds=.2, idle_seconds=3., trim_seconds=1.)
    calls = []

    class Proc:
        returncode = 0
        stderr = ""
        def __init__(self, stdout): self.stdout = stdout

    def fake_run(cmd, **kwargs):
        kernel = cmd[cmd.index("--kernel") + 1]
        size = int(cmd[cmd.index("--problem-size") + 1])
        block = int(cmd[cmd.index("--block-size") + 1])
        calls.append((size, block, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        if block == 512 or size == 32:
            return SimpleNamespace(returncode=1, stdout="", stderr="verification failure")
        raw = {"kernel": kernel, "problem_size": size, "block_size": block, "warmups": 1,
               "repeats": 2, "verify": True, "verification_passed": True,
               "device": {"uuid": "GPU-test"}, "bytes_moved": 10, "flops": 0,
               "runtime_ms_samples": [1., 1.1]}
        return Proc(json.dumps(raw))

    captures = []
    def fake_capture(binary_arg, scenario, out, device, args_arg, seed, trials):
        confirming = "confirmations" in out.parts
        captures.append((scenario.block_size, seed, trials, device, confirming))
        launch_count = {(128, False): 500, (256, False): 480,
                        (128, True): 500, (256, True): 470}[scenario.block_size, confirming]
        energy = 1.0 if scenario.block_size == 128 else .9
        return [{"trial": i, "kernel": scenario.kernel, "problem_size": scenario.problem_size,
                 "block_size": scenario.block_size, "verification_passed": True,
                 "estimated_j_per_launch": energy, "sustained_duration_s": 5.,
                 "launches_per_second": launch_count / 5., "mean_power_w": 100.,
                 "window": {"launches": launch_count}} for i in (1, 2)]

    monkeypatch.setattr(autotune.subprocess, "run", fake_run)
    monkeypatch.setattr(autotune, "query_gpu", lambda device: {"gpu_uuid": "GPU-test"})
    monkeypatch.setattr(autotune, "_capture", fake_capture)
    assert autotune.run_autotune(args) == 2
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert len(report["candidates"]) == 2
    assert report["missing_requested_workloads"] == ["vector_add:32"]
    assert report["status"] == "unconfirmed"
    assert any(x["block_size"] == 512 for x in report["disqualified_candidates"])
    assert report["selection"]["vector_add:16"]["selected_candidate_id"].endswith("b256")
    assert report["selected_confirmed"]["vector_add:16"] is False
    selected_confirmation = next(x for x in report["confirmations"] if x["candidate_id"].endswith("b256"))
    assert selected_confirmation["confirmation_passed"] is True  # within 20% of its own first run
    assert selected_confirmation["speed_constraint_passed"] is False  # outside 5% of confirmed fastest
    assert len(captures) == 4  # both Pareto finalists get fresh confirmation trials
    measured = [x for x in captures if not x[4]]
    confirmed = [x for x in captures if x[4]]
    assert {x[0] for x in measured} == {128, 256}
    assert {x[0] for x in confirmed} == {128, 256}
    assert len({x[1] for x in measured + confirmed}) == 4
    assert all(x[3] == "GPU-test" for x in captures)
    assert {block for _, block, _ in calls} == {128, 256, 512}
    assert all(device == "GPU-test" for _, _, device in calls)


def test_workflow_writes_failure_report_when_no_candidate_survives(tmp_path, monkeypatch):
    binary = tmp_path / "bench"
    binary.write_text("fixture")
    args = Namespace(binary=str(binary), kernels=["vector_add"], problem_sizes=[8], block_sizes=[128],
        outdir=str(tmp_path / "failed"), device="0", trials=2, confirm_trials=2,
        speed_tolerance=.05, seed=1, warmups=1, repeats=2, timeout_seconds=5,
        duration_seconds=5., interval_seconds=.2, idle_seconds=3., trim_seconds=1.)
    monkeypatch.setattr(autotune, "query_gpu", lambda _: {"gpu_uuid": "GPU-test"})
    monkeypatch.setattr(autotune.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=1, stdout="preflight out", stderr="correctness failed"))
    assert autotune.run_autotune(args) == 2
    report = json.loads((tmp_path / "failed" / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["selection"] == {}
    assert "correctness failed" in report["disqualified_candidates"][0]["reason"]
