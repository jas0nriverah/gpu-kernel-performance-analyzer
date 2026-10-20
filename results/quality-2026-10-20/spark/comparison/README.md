# H200 → DGX Spark timing comparisons

Existing comparison CLI, mean sample runtime, default 5% slowdown threshold and within-run CV limit 0.1. All 74 scenarios matched in each corresponding trial; no scenarios were added or missing.

| Trial | Scenarios >5% slower on Spark | Noisy scenarios | Median scenario runtime change (%) |
| --- | ---: | ---: | ---: |
| 1 | 66 | 2 | 244.306 |
| 2 | 66 | 3 | 243.023 |
| 3 | 67 | 4 | 244.331 |

Positive change means slower Spark runtime. The median summarizes scenario-level percentages, not a total workload runtime or aggregate speedup. Corresponding trial numbers are independent captures, not paired randomized experiments. Every comparison is ineligible for a passing CI gate because GPU metadata differs. Host architecture, GPU design, memory, clocks, compiler/toolkit, capture time and system conditions differ; these ratios cannot isolate a hardware effect or establish statistical significance. CV flags descriptive within-run noise. These reports use timing only; they make no power or energy claim.

Each trial directory preserves `comparison.json`, `comparison.csv`, and `REPORT.md`. `summary.json` retains counts, range/median changes, gate reasons and input manifest/raw-sample hashes.

Reproduce with a fresh output directory:

```bash
.venv/bin/python -m gpu_kernel_analyzer compare \
  --baseline outputs/prizepicks-h200-2026-10-20/timing/trial-1 \
  --candidate outputs/quality-2026-10-20/spark/timing/trial-1 \
  --allow-device-mismatch --statistic mean \
  --outdir outputs/quality-replay/h200-spark-trial-1
```

Repeat for trial 2 and trial 3. No `--fail-on-regression` flag is used for these descriptive cross-device comparisons.
