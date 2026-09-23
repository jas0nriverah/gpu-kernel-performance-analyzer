# Measured energy-aware autotuning

`gpu-kernel-analyzer autotune run` searches existing benchmark kernel families and launch block sizes. It first runs each candidate with correctness verification enabled on the requested GPU, then captures sustained board power and launch throughput using the power measurement protocol. Candidates that fail correctness, telemetry validation, or complete trial coverage are disqualified.

For every mathematical workload and problem size, the tuner finds the fastest measured median runtime per launch, keeps candidates within the configured speed tolerance (5% by default), and selects the one with the lowest median estimated joules per launch. GEMM implementations of the same size can be compared; unrelated kernels and different sizes are grouped separately. The report includes the speed/energy Pareto frontier so the tradeoff remains visible.

The tuner independently repeats every speed-eligible candidate and the original fastest candidate, with separate capture runs. It requires the confirmation medians to agree with each candidate's first run within 20%, then checks that the selected candidate remains within the configured speed tolerance of the confirmed fastest candidate and remains the unique energy minimum. It marks candidates indistinguishable when their energy median gap is no larger than the sum of their median absolute deviations in either discovery or confirmation trials. This is a descriptive dispersion screen, not a significance test; tied candidates yield no unique energy claim and the command exits nonzero. A missing workload or incomplete capture also makes the command exit nonzero while preserving its report and raw evidence. Median absolute deviations, raw trial data, worker output, telemetry CSVs, capture manifests, and artifact hashes are retained. Board power includes device circuitry, and GPU sensor averaging plus a small number of trials limits precision. Candidate trials within each capture are contiguous; candidate order is randomized, but the seed does not randomize a multi-workload sequence within a single capture. No model prediction participates in ranking.

Example:

```bash
gpu-kernel-analyzer autotune run \
  --binary build/gpu_benchmark \
  --kernels vector_add \
  --problem-sizes 1048576 \
  --block-sizes 128 256 512 1024 \
  --outdir results/vector-add-autotune \
  --device 0
```

The output directory must not already exist. `report.md` summarizes selected configurations and Pareto tradeoffs; `report.json` contains the full ranking evidence. Per-candidate folders retain correctness-worker and sustained power captures. Use one kernel family and one problem size for a focused search; GEMM naive and tiled kernels may be supplied together when comparing the same matrix size. Existing fixed-block restrictions, such as the tiled GEMM tile size, are enforced.

The CPU tests mock CUDA execution and power capture. Actual selection quality depends on an idle GPU, adequate trial duration, and comparable thermal and clock conditions across candidates. An H100 example, including source hashes, raw captures and the follow-up uncertainty review, is available in [the autotuning evidence directory](../results/h100-autotune-2026-09-22/). Its original search selected block 256. Follow-up measurements made block 256 the fastest practical recommendation among the speed-eligible candidates, while the prespecified discovery dispersion screen leaves its unique energy advantage unconfirmed.
