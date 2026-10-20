# Benchmark comparison

NVIDIA H200 → NVIDIA GH200 480GB

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 6.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 3 matched scenarios exceed the CV limit (0.1).
- Gate: 6 matched scenarios exceed the slowdown threshold.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008548 | 0.007586 | 1.13× | -11.3% | improvement | no |
| gemm_naive | 256 | 16 | 0.013544 | 0.012559 | 1.08× | -7.3% | improvement | no |
| gemm_naive | 512 | 16 | 0.060570 | 0.059505 | 1.02× | -1.8% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.418335 | 0.417867 | 1.00× | -0.1% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.178168 | 3.178554 | 1.00× | +0.0% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007583 | 0.006377 | 1.19× | -15.9% | improvement | yes |
| gemm_tiled | 256 | 16 | 0.011635 | 0.010618 | 1.10× | -8.7% | improvement | no |
| gemm_tiled | 512 | 16 | 0.042013 | 0.041225 | 1.02× | -1.9% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279112 | 0.278226 | 1.00× | -0.3% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.106191 | 2.106502 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014721 | 0.013888 | 1.06× | -5.7% | improvement | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009916 | 0.008813 | 1.13× | -11.1% | improvement | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007322 | 0.006316 | 1.16× | -13.7% | improvement | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006496 | 0.005615 | 1.16× | -13.6% | improvement | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044360 | 0.043580 | 1.02× | -1.8% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024550 | 0.023994 | 1.02× | -2.3% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.014889 | 0.014072 | 1.06× | -5.5% | improvement | no |
| memcpy_bandwidth | 4194304 | 512 | 0.012331 | 0.011247 | 1.10× | -8.8% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162598 | 0.162022 | 1.00× | -0.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084113 | 0.083620 | 1.01× | -0.6% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.054352 | 0.056728 | 0.96× | +4.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 512 | 0.058319 | 0.060920 | 0.96× | +4.5% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635093 | 0.634529 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.321023 | 0.320324 | 1.00× | -0.2% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.201139 | 0.213290 | 0.94× | +6.0% | regression | no |
| memcpy_bandwidth | 67108864 | 512 | 0.217191 | 0.229617 | 0.95× | +5.7% | regression | no |
| reduction | 1048576 | 64 | 0.015455 | 0.014476 | 1.07× | -6.3% | improvement | no |
| reduction | 1048576 | 128 | 0.010454 | 0.009525 | 1.10× | -8.9% | improvement | no |
| reduction | 1048576 | 256 | 0.009491 | 0.008616 | 1.10× | -9.2% | improvement | no |
| reduction | 1048576 | 512 | 0.010005 | 0.009178 | 1.09× | -8.3% | improvement | yes |
| reduction | 4194304 | 64 | 0.044938 | 0.044155 | 1.02× | -1.7% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025323 | 0.024354 | 1.04× | -3.8% | within_threshold | no |
| reduction | 4194304 | 256 | 0.022025 | 0.021364 | 1.03× | -3.0% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024013 | 0.022998 | 1.04× | -4.2% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163028 | 0.162397 | 1.00× | -0.4% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084725 | 0.083976 | 1.01× | -0.9% | within_threshold | no |
| reduction | 16777216 | 256 | 0.080714 | 0.080588 | 1.00× | -0.2% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091401 | 0.090646 | 1.01× | -0.8% | within_threshold | no |
| reduction | 67108864 | 64 | 0.635498 | 0.634935 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321309 | 0.320898 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 256 | 0.305892 | 0.306423 | 1.00× | +0.2% | within_threshold | no |
| reduction | 67108864 | 512 | 0.347774 | 0.346825 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014896 | 0.013757 | 1.08× | -7.6% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009918 | 0.008913 | 1.11× | -10.1% | improvement | no |
| stencil_1d | 1048576 | 256 | 0.007478 | 0.006333 | 1.18× | -15.3% | improvement | no |
| stencil_1d | 1048576 | 512 | 0.006979 | 0.005823 | 1.20× | -16.6% | improvement | no |
| stencil_1d | 4194304 | 64 | 0.044415 | 0.043448 | 1.02× | -2.2% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024765 | 0.023942 | 1.03× | -3.3% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015068 | 0.014038 | 1.07× | -6.8% | improvement | no |
| stencil_1d | 4194304 | 512 | 0.013769 | 0.012536 | 1.10× | -9.0% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162627 | 0.162180 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084244 | 0.083581 | 1.01× | -0.8% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.059092 | 0.061414 | 0.96× | +3.9% | within_threshold | no |
| stencil_1d | 16777216 | 512 | 0.063089 | 0.065292 | 0.97× | +3.5% | within_threshold | no |
| stencil_1d | 67108864 | 64 | 0.635024 | 0.634743 | 1.00× | -0.0% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321463 | 0.320671 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.220058 | 0.231048 | 0.95× | +5.0% | within_threshold | no |
| stencil_1d | 67108864 | 512 | 0.236023 | 0.247301 | 0.95× | +4.8% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014755 | 0.013761 | 1.07× | -6.7% | improvement | no |
| vector_add | 1048576 | 128 | 0.009835 | 0.009146 | 1.08× | -7.0% | improvement | yes |
| vector_add | 1048576 | 256 | 0.007455 | 0.006316 | 1.18× | -15.3% | improvement | no |
| vector_add | 1048576 | 512 | 0.006644 | 0.005892 | 1.13× | -11.3% | improvement | no |
| vector_add | 4194304 | 64 | 0.044505 | 0.044271 | 1.01× | -0.5% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025014 | 0.024456 | 1.02× | -2.2% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.019414 | 0.019202 | 1.01× | -1.1% | within_threshold | no |
| vector_add | 4194304 | 512 | 0.020371 | 0.020127 | 1.01× | -1.2% | within_threshold | no |
| vector_add | 16777216 | 64 | 0.162869 | 0.162412 | 1.00× | -0.3% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084440 | 0.083837 | 1.01× | -0.7% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.063150 | 0.067726 | 0.93× | +7.2% | regression | no |
| vector_add | 16777216 | 512 | 0.066689 | 0.070523 | 0.95× | +5.7% | regression | no |
| vector_add | 67108864 | 64 | 0.635110 | 0.634925 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321651 | 0.320718 | 1.00× | -0.3% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.236536 | 0.256479 | 0.92× | +8.4% | regression | no |
| vector_add | 67108864 | 512 | 0.250788 | 0.268762 | 0.93× | +7.2% | regression | no |
