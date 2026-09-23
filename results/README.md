# Captured benchmark evidence

- [`a100-2026-04-28/`](a100-2026-04-28/): original 12-scenario A100 capture recovered from the earlier local checkout. Four original files are copied without rewriting values or the manifest. The historical schema lacks the newer min/max summary columns; use `compare` to validate its raw timing consistency rather than the current strict artifact-schema command.
- [`h100-2026-09-22/`](h100-2026-09-22/): baseline, three 74-scenario sweeps, 18 boundary checks, and two separate profiler captures.
- [`h100-autotune-2026-09-22/`](h100-autotune-2026-09-22/): correctness-gated H100 energy-aware vector-add search across 128/256/512-thread blocks. [`RUN.md`](h100-autotune-2026-09-22/RUN.md) records original selection, command and provenance; [`energy-review.md`](h100-autotune-2026-09-22/energy-review.md) records the b512 follow-up and final dispersion-limited outcome. `report.json` and nested captures retain raw telemetry and worker evidence.
- [`comparisons/`](comparisons/): generated cross-GPU and independent H100 trial comparisons. These are descriptive reports; inspect noise and compatibility flags before using a CI decision.

Every scenario in these timing captures requested correctness verification. Raw CUDA-event samples are retained; generated charts and aggregate tables can be rebuilt with `scripts/summarize_results.py`. See the [results discussion](../docs/real_gpu_validation.md) for limits and interpretation.

Manifests preserve capture-time paths and source hashes. H100 runs were captured from a working tree based on the recorded git commit, with the CUDA verification/build changes included in `h100-2026-09-22/benchmark-source.patch`. The base git commit alone does not describe those uncommitted changes. Python orchestration was being extended during this work; each run records its own source hashes. Published kernel timing code and scenario configurations were unchanged across the three sweeps.
