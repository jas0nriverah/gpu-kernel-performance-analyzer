# Engineering decisions

The project is a local batch system with a CUDA execution worker and a Python control layer. Its useful backend properties are explicit contracts, bounded execution, traceable artifacts, and automated regression checks.

## Execution boundary

Python validates a scenario configuration, expands it into jobs, and launches one subprocess per scenario. The CUDA worker owns allocations, warmups, event timing, and correctness checks. Its final stdout line is a JSON result; stderr carries diagnostics.

The runner checks that the response matches the requested kernel, dimensions, block size, warmups, repeats, and verification flag. It rejects unsuccessful verification, missing samples, and nonpositive or nonfinite timings. Each process has a configurable 120-second timeout. A failure stops the sweep before summary artifacts are written. A nonempty output directory is rejected to protect previous evidence. `gpu_benchmark --device-info` reports metadata and the selected CUDA device UUID without running a kernel; the UUID is included in worker results for device identity checks.

This boundary lets CPU-only tests replace the CUDA executable with a deterministic fixture. It also lets the analysis pipeline run without a CUDA installation.

## Artifact contract

| Artifact | Responsibility |
| --- | --- |
| `run_manifest.json` | Run identity, GPU metadata, source/binary/config hashes, environment, profiler state |
| `timing_samples.csv` | Every measured event duration, keyed by run, scenario, and sample index |
| `benchmark_summary.csv` | Runtime distribution and derived throughput per scenario |
| `metrics_provenance.csv` | Distinguishes measured values, estimates, and unavailable counters |
| `comparison.json` / `.csv` | Machine-readable scenario matches, deltas, noise flags, missing coverage, gate decision |

Flat files keep a small experiment portable and inspectable. A database or queue would add operating cost without helping the current single-GPU workflow. Concurrent workers would require explicit scheduling and resource ownership because simultaneous workloads distort timings.

## Regression policy

`compare` joins on `(kernel, problem_size, block_size)`. It independently reconstructs mean, median, and coefficient of variation from raw samples, checking run IDs, sample indices, repeat counts, and verification status. This also supports the historical A100 schema, which predates min/max columns.

Median is the default comparison statistic; mean is available for compatibility with throughput calculations. A slowdown greater than the configured percentage is a regression. Within-run CV above the configured limit marks a result noisy. Neither a percentage threshold nor CV is a statistical significance test.

With `--fail-on-regression`, the command returns:

| Exit code | Meaning |
| --- | --- |
| `0` | Every matched scenario passes; baseline coverage is preserved |
| `1` | Regression, excessive noise, missing baseline scenario, or changed GPU/warmup/repeat protocol |
| `2` | Invalid inputs, inconsistent samples, or incompatible comparison without explicit opt-in |

Different GPUs require `--allow-device-mismatch` and are always ineligible for a passing CI gate. Code/binary/environment changes are exposed in the report. Same-device identity means matching device metadata, not proof of identical clocks or host load: use a dedicated runner and inspect the environment when interpreting changes.

## Correctness and launch bounds

The six kernels use deterministic seeded nonuniform input data and CPU-reference checks. Verification covers every vector, copy, stencil, and GEMM output, plus every reduction block partial. Small GEMMs use arbitrary dense inputs and full CPU dot products; larger GEMMs use a structured rank-one input with a closed-form reference, keeping CPU verification O(N²). This exercises every output and tail while reducing large-N input diversity. Tolerances combine absolute and relative error and reject nonfinite values. See [oracle coverage and H100 validation](correctness.md).

Scenario parsing and native worker entry points both reject unsafe sizes and invalid launch shapes. Checks include reduction's power-of-two block requirement, per-block thread limits, CUDA grid dimensions, and signed GEMM index bounds. Native limits come from the currently selected device properties and are checked before allocations.

## Testing boundaries

CPU tests cover parsing, subprocess failures and timeouts, malformed responses, statistics, metric provenance, report generation, and regression policy. Real GPU validation covers compilation, six kernels, nonmultiple/edge dimensions, repeated sweeps, and optional profiler capture. CPU CI cannot prove CUDA correctness or performance.

## Sweep recovery and tuning

Long sweeps can save atomic per-scenario checkpoints and resume after a process interruption. Resume validates the scenario/config fingerprint and selected GPU UUID before reusing results; it refuses mismatched or corrupted state. See [sweep resumability](resumability.md).

The optional block-size tuner performs a small correctness-checked candidate sweep and records its choice separately from the fixed benchmark configurations. It is an experiment helper, not a replacement for repeated, controlled performance studies. See [autotuning](autotuning.md).

## Remaining engineering priorities

1. **Controlled performance CI.** Run identical suites on a dedicated GPU runner with recorded clocks, temperature, power, compiler flags, and multiple randomized independent trials. Hardware availability should remain separate from portable unit tests.
2. **Model evaluation by held-out workload.** The optional ridge model is exploratory. Evaluate by held-out sizes/runs and report a simple baseline before using predictions to drive tuning. Do not pool A100/H100 training rows: the current features do not encode GPU identity.

These are follow-up improvements, not implemented features. A REST API, scheduler, or database becomes useful when there are multiple users or remote workers; it is not required to demonstrate the current system's behavior.
