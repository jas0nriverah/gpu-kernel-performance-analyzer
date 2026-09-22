# Comparing runs and gating regressions

Compare two independently captured runs on the same GPU:

```bash
python -m gpu_kernel_analyzer compare \
  --baseline outputs/before --candidate outputs/after \
  --outdir outputs/comparison --threshold-pct 5 --max-cv 0.10 \
  --fail-on-regression
```

The output contains `REPORT.md`, `comparison.csv`, and `comparison.json`. By default, runtime medians determine changes. A positive change means slower; speedup is baseline runtime divided by candidate runtime. `--statistic mean` compares means instead.

A CI gate fails for regressions, missing baseline scenarios, excessive within-run variability, or different GPU/warmup/repeat settings. Invalid inputs return exit code 2. A descriptive comparison without `--fail-on-regression` returns 0 after successfully writing a report, even when regressions exist. See [the policy](engineering.md#regression-policy).

## Explore the published evidence without a GPU

```bash
python -m gpu_kernel_analyzer compare \
  --baseline results/a100-2026-04-28 \
  --candidate results/h100-2026-09-22/baseline \
  --allow-device-mismatch --statistic mean \
  --outdir outputs/a100-h100
```

This cross-device report is descriptive and cannot pass a performance CI gate. The historical A100 capture used an older source revision and toolchain; the ratio is not an isolated measure of the architecture improvement.

To inspect repeatability on the H100:

```bash
python -m gpu_kernel_analyzer compare \
  --baseline results/h100-2026-09-22/trial-1 \
  --candidate results/h100-2026-09-22/trial-2 \
  --outdir outputs/h100-repeatability
```

The comparator requires raw timings and successful verification. It rejects duplicate scenarios, missing/duplicate timing indices, NaN/zero/negative timings, run identity mismatches, and summaries that disagree with the raw samples. Historical files without min/max columns remain readable.
