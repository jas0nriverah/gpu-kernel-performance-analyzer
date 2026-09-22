# Benchmark comparison

NVIDIA A100 80GB PCIe → NVIDIA H100 80GB HBM3

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 12; missing: 0; added: 12; regressions: 0.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 1 matched scenarios exceed the CV limit (0.1).
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: git_commit.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: scenario_file_sha256.
- Run metadata differs: runtime_environment.
- Run metadata differs: source_sha256.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 256 | 16 | 0.021777 | 0.013850 | 1.57× | -36.4% | improvement | no |
| gemm_naive | 512 | 16 | 0.103697 | 0.060557 | 1.71× | -41.6% | improvement | no |
| gemm_tiled | 256 | 16 | 0.017818 | 0.011838 | 1.51× | -33.6% | improvement | no |
| gemm_tiled | 512 | 16 | 0.069905 | 0.042255 | 1.65× | -39.6% | improvement | no |
| reduction | 1048576 | 128 | 0.013824 | 0.010542 | 1.31× | -23.7% | improvement | no |
| reduction | 1048576 | 256 | 0.013619 | 0.009626 | 1.41× | -29.3% | improvement | no |
| reduction | 4194304 | 128 | 0.035806 | 0.025425 | 1.41× | -29.0% | improvement | no |
| reduction | 4194304 | 256 | 0.035226 | 0.022028 | 1.60× | -37.5% | improvement | no |
| vector_add | 1048576 | 128 | 0.011708 | 0.009964 | 1.18× | -14.9% | improvement | no |
| vector_add | 1048576 | 256 | 0.009045 | 0.007462 | 1.21× | -17.5% | improvement | yes |
| vector_add | 4194304 | 128 | 0.037922 | 0.025180 | 1.51× | -33.6% | improvement | no |
| vector_add | 4194304 | 256 | 0.037854 | 0.023261 | 1.63× | -38.6% | improvement | no |

## Added in candidate

- `memcpy_bandwidth`, size 1048576, block 256
- `memcpy_bandwidth`, size 4194304, block 256
- `reduction`, size 1048576, block 64
- `reduction`, size 1048576, block 512
- `reduction`, size 4194304, block 64
- `reduction`, size 4194304, block 512
- `stencil_1d`, size 1048576, block 256
- `stencil_1d`, size 4194304, block 256
- `vector_add`, size 1048576, block 64
- `vector_add`, size 1048576, block 512
- `vector_add`, size 4194304, block 64
- `vector_add`, size 4194304, block 512
