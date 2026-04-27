# GPU Kernel Performance Analyzer - Strict Audit

## Pass/Fail Summary

Overall verdict: **FAIL (for strict interview/demo readiness right now)**  
Reason: core MVP flow works, but there are credibility and command-surface issues that must be fixed before a real demo.

### Category summary

- **Core non-CUDA pipeline (`sweep -> validate -> analyze -> report`)**: PASS (with fixture binary)
- **Schema/provenance enforcement**: PASS with caveats
- **README/docs alignment**: PARTIAL PASS
- **Requested CLI command surface (`python -m gpu_kernel_analyzer.cli ...`)**: **FAIL**
- **Optional Nsight behavior non-blocking**: PASS
- **Technical credibility under skeptical review**: **FAIL until critical/major issues below are fixed**

---

## Exact Commands Run

1. `python -m pytest -v`  
   - Result: PASS  
   - Output summary: 9 tests collected, 9 passed.

2. `python -m gpu_kernel_analyzer.cli --help`  
   - Result: **FAIL behavior** (exit code 0, no output)

3. `python -m gpu_kernel_analyzer.cli profile ncu-detect`  
   - Result: **FAIL behavior** (exit code 0, no output)

4. `python -m gpu_kernel_analyzer.cli analyze --help`  
   - Result: **FAIL behavior** (exit code 0, no output)

5. `python -m gpu_kernel_analyzer --help`  
   - Result: PASS  
   - Output summary: proper argparse help with commands `benchmark/artifacts/analyze/profile`.

6. `python -m gpu_kernel_analyzer profile ncu-detect`  
   - Result: PASS  
   - Output:
     - `available: false`
     - `enabled: false`
     - `status: not_detected`

7. `python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/final_smoke`  
   - Result: PASS  
   - Output summary: validation passed.

8. `python -m gpu_kernel_analyzer analyze full --run-dir outputs/final_smoke`  
   - Result: PASS  
   - Output summary: heuristics printed for all scenarios, `REPORT.md` and plots generated.

9. `nvcc --version`  
   - Result: NVCC NOT AVAILABLE  
   - Output summary: command not found.

10. CMake configure/build  
   - Result: NOT RUN (per instruction, only if `nvcc` available).

---

## Files Reviewed

Reviewed all created/modified implementation surfaces:

- Benchmark/CUDA: `benchmarks/CMakeLists.txt`, all files under `benchmarks/include` and `benchmarks/src` (including all kernels).
- Python package: all files under `src/gpu_kernel_analyzer`.
- Configs: `configs/benchmark_scenarios.yaml`, `configs/ncu_metric_sets.yaml`, `configs/benchmark_scenarios.json`.
- Docs/metadata: `README.md`, `pyproject.toml`, all files under `docs/`, `.github/workflows/ci.yml`, `.gitignore`, `LICENSE`.
- Tests: all files under `tests/` including fixture script.
- Generated artifacts used for consistency checks: `outputs/final_smoke/*` including CSV/JSON/`REPORT.md`.

---

## Issues Found

## 1) `python -m gpu_kernel_analyzer.cli ...` does nothing (silent no-op)
- **Severity:** major
- **Evidence:** all requested `python -m gpu_kernel_analyzer.cli ...` commands returned exit 0 with no output.
- **Technical cause:** `src/gpu_kernel_analyzer/cli.py` defines `main()` but has no module execution guard (`if __name__ == "__main__": main()`).
- **Impact:** interviewer/user can think commands succeeded when they actually did nothing; undermines trust and operability checks.
- **Recommended fix:** add module execution guard in `cli.py` and test it.

## 2) Nsight metric import can mark arbitrary CSV values as “measured”
- **Severity:** critical
- **Evidence:** `src/gpu_kernel_analyzer/ncu.py` imports any CSV with columns `kernel,metric_name,metric_value`, then sets `status=measured` and updates manifest to `parsed_metrics`.
- **Impact:** measured profiler claims can be fabricated or accidentally incorrect without provenance to actual Nsight command/output.
- **Recommended fix:** require and store stronger provenance (tool command, raw report hash, timestamp, optionally parser signature), and reject imports missing required Nsight metadata.

## 3) Nsight import applies kernel-level values across all problem sizes/blocks
- **Severity:** major
- **Evidence:** matching in `ncu.py` uses only `kernel` + `metric_name`, not problem size/block dimensions.
- **Impact:** scenario-specific profiler metrics may be misapplied broadly, leading to misleading analysis claims.
- **Recommended fix:** include scenario identifiers (`problem_size`, `block_size`, possibly run_id) in import contract and matching.

## 4) CLI help text mismatches docs on scenario format
- **Severity:** minor
- **Evidence:** `cli.py` help says `--scenarios` is “Path to benchmark scenarios JSON”; README/docs use YAML and loader supports YAML+JSON.
- **Impact:** user confusion; reviewer perceives inconsistency.
- **Recommended fix:** update help text to “JSON or YAML”.

## 5) JSON serialization in C++ benchmark output is not escaped
- **Severity:** minor
- **Evidence:** `benchmarks/src/benchmark_runner.cu` manually emits JSON strings without escaping device/kernel strings.
- **Impact:** uncommon but possible invalid JSON if strings contain special characters; parser fragility.
- **Recommended fix:** use robust JSON library or escape string fields explicitly.

## 6) Test suite is useful but still mostly fixture/smoke-level for benchmark integrity
- **Severity:** minor
- **Evidence:** tests validate schema/CLI/analysis paths using fixture data; there is no real CUDA binary contract test in CI and no test covering `.cli` module invocation.
- **Impact:** regressions in real CUDA runtime path may go undetected; operability bug already slipped through.
- **Recommended fix:** add tests for module invocation and, when GPU CI unavailable, at least stricter contract tests against canonical benchmark JSON samples from real binary runs.

---

## Schema Consistency Check

Result: **mostly consistent** across benchmark output -> validation -> analysis -> report.

- `timing_samples.csv`, `benchmark_summary.csv`, `metrics_provenance.csv`, and `run_manifest.json` are structurally coherent with validator expectations.
- `analyze full` and `REPORT.md` are generated from artifact files (not hardcoded sample text).
- `REPORT.md` truncates tables to first 10 rows by design; this is acceptable but should be understood during review.
- Profiler metrics policy is enforced conditionally via manifest `nsight_compute.status == parsed_metrics`.

---

## README/DOCS vs Implementation

Result: **mostly aligned**, with these caveats:

- README commands using `python -m gpu_kernel_analyzer ...` work.
- Requested `python -m gpu_kernel_analyzer.cli ...` does not work (major issue above).
- CUDA-unavailable behavior is documented and observed (`nvcc` missing).
- Optional Nsight integration is non-blocking and documented.
- Claim “never fabricate counters” is not fully defensible while arbitrary CSV import can produce `measured` profiler rows.

---

## Is this currently resume-defensible?

**Not yet for a strict NVIDIA-style performance interview.**

Why:
- A critical trust gap exists in profiler metric ingestion/provenance.
- A major CLI operability bug exists for module-level invocation.
- Real CUDA build/run path could not be validated in this environment (`nvcc` unavailable), so only fixture-backed evidence exists here.

---

## Must Fix Before Real Demo

1. Fix `python -m gpu_kernel_analyzer.cli ...` execution behavior and add a test.
2. Harden Nsight metric provenance so “measured” profiler metrics cannot be asserted from arbitrary CSVs without traceable source evidence.
3. Make Nsight import scenario-specific (`kernel + problem_size + block_size` at minimum).
4. Resolve CLI help/docs mismatch for scenario format (JSON vs YAML).
5. Add one demo validation run with real CUDA binary artifacts (when `nvcc`/GPU available), and include those artifacts as verifiable evidence.

---

## Post-fix validation (critical/major scope)

### Commands run after fixes

1. `python -m pytest -v`
   - Result: PASS
   - Summary: 17 collected, 17 passed.

2. `python -m gpu_kernel_analyzer.cli --help`
   - Result: PASS
   - Summary: argparse help prints correctly with command tree.

3. `python -m gpu_kernel_analyzer.cli benchmark --help`
   - Result: PASS
   - Summary: benchmark subcommand help prints correctly.

4. `python -m gpu_kernel_analyzer.cli profile ncu-detect`
   - Result: PASS
   - Summary: emits JSON with `{available, enabled, path, status}`.

5. `python -m gpu_kernel_analyzer.cli analyze --help`
   - Result: PASS
   - Summary: module-level analyze help prints correctly.

6. Non-CUDA README demo commands:
   - `python -m gpu_kernel_analyzer benchmark sweep --binary tests/fixtures/fake_benchmark.py --binary-interpreter python --scenarios configs/benchmark_scenarios.yaml --outdir outputs/sample_fixture`
   - `python -m gpu_kernel_analyzer artifacts validate --run-dir outputs/sample_fixture`
   - `python -m gpu_kernel_analyzer analyze full --run-dir outputs/sample_fixture`
   - Result: PASS
   - Summary: fixture artifacts, report, and plots generated successfully.

7. Hardened Nsight import command check:
   - `python -m gpu_kernel_analyzer.cli profile ncu-import --run-dir outputs/sample_fixture --source-tool ncu --source-file sample_ncu_raw.csv --metric-set default_profiler_set --ncu-csv outputs/sample_fixture/ncu_import_sample.csv`
   - Result: PASS
   - Summary: imported one scenario-specific metric and post-import validation passed.

### Critical/major issue status

- **CLI module invocation no-op (`python -m gpu_kernel_analyzer.cli`)**: RESOLVED
  - Added executable module path (`if __name__ == "__main__": main()`) and direct invocation tests.

- **Arbitrary Nsight CSV import measured-claim risk**: RESOLVED
  - Import now requires provenance metadata (`source_tool`, `source_file`, `metric_set`) and stamps import timestamp.
  - Validator now rejects measured profiler rows lacking traceable provenance tokens/manifest metadata.

- **Nsight import not scenario-specific**: RESOLVED
  - Import now requires and matches `kernel + problem_size + block_size + metric_name`.
  - Ambiguous or missing matches fail with clear errors.

### Remaining limitations (non-critical/non-major)

- Real CUDA build/run remains unverified in this environment (`nvcc` unavailable).
- Real Nsight workflow with native `ncu` output still needs validation on an actual NVIDIA GPU system.
- Manual JSON serialization in C++ benchmark output remains a minor robustness risk if string escaping edge-cases occur.


