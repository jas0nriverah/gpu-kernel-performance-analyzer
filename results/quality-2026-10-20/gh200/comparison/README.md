# Descriptive timing comparisons

Generated with the existing `gpu_kernel_analyzer compare` command, `--allow-device-mismatch --statistic mean`, default slowdown threshold 5% and CV threshold 0.1. Each corresponding trial pair matches 74 scenarios. Unsuffixed directories contain trial 1; `-trial-2` and `-trial-3` contain those trials. Every directory retains `comparison.json`, `comparison.csv` and `REPORT.md`; `summary.json` extracts counts and descriptive changes.

| Baseline → candidate | Trial 1 / 2 / 3 scenarios >5% slower | Trial 1 / 2 / 3 median change (%) |
| --- | --- | --- |
| H100 → H200 | 0 / 0 / 0 | -0.071 / -0.453 / -0.319 |
| H100 → GH200 | 0 / 0 / 0 | -5.022 / -5.077 / -5.008 |
| H200 → GH200 | 8 / 6 / 9 | -1.223 / -0.987 / -1.010 |

Positive change means slower candidate runtime. Medians summarize scenario-level percentage changes; they are not an aggregate workload speedup. Corresponding trial numbers are independent captures, not paired randomized treatments. Every comparison is ineligible for a passing CI gate because device metadata differs. Source/binary, host architecture, toolkit, date and environment differences confound hardware attribution. CV flags within-run noise; this is not a statistical significance test.

Example reproduction (use a fresh destination):

```bash
.venv/bin/python -m gpu_kernel_analyzer compare \
  --baseline outputs/prizepicks-h200-2026-10-20/timing/trial-1 \
  --candidate outputs/quality-2026-10-20/gh200/timing/trial-1 \
  --allow-device-mismatch --statistic mean \
  --outdir outputs/quality-replay/h200-gh200
```
