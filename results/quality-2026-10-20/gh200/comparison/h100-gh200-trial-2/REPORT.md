# Benchmark comparison

NVIDIA H100 80GB HBM3 → NVIDIA GH200 480GB

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 0.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 3 matched scenarios exceed the CV limit (0.1).
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: git_commit.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.
- Run metadata differs: source_sha256.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008813 | 0.007586 | 1.16× | -13.9% | improvement | no |
| gemm_naive | 256 | 16 | 0.013612 | 0.012559 | 1.08× | -7.7% | improvement | no |
| gemm_naive | 512 | 16 | 0.060358 | 0.059505 | 1.01× | -1.4% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.417903 | 0.417867 | 1.00× | -0.0% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.177004 | 3.178554 | 1.00× | +0.0% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007628 | 0.006377 | 1.20× | -16.4% | improvement | no |
| gemm_tiled | 256 | 16 | 0.011645 | 0.010618 | 1.10× | -8.8% | improvement | no |
| gemm_tiled | 512 | 16 | 0.042056 | 0.041225 | 1.02× | -2.0% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279035 | 0.278226 | 1.00× | -0.3% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.105848 | 2.106502 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014764 | 0.013888 | 1.06× | -5.9% | improvement | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009858 | 0.008813 | 1.12× | -10.6% | improvement | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007512 | 0.006316 | 1.19× | -15.9% | improvement | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006662 | 0.005615 | 1.19× | -15.7% | improvement | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044412 | 0.043580 | 1.02× | -1.9% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024829 | 0.023994 | 1.03× | -3.4% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015075 | 0.014072 | 1.07× | -6.7% | improvement | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013793 | 0.011247 | 1.23× | -18.5% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162644 | 0.162022 | 1.00× | -0.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084159 | 0.083620 | 1.01× | -0.6% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060468 | 0.056728 | 1.07× | -6.2% | improvement | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064780 | 0.060920 | 1.06× | -6.0% | improvement | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635264 | 0.634529 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320990 | 0.320324 | 1.00× | -0.2% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225416 | 0.213290 | 1.06× | -5.4% | improvement | no |
| memcpy_bandwidth | 67108864 | 512 | 0.242744 | 0.229617 | 1.06× | -5.4% | improvement | no |
| reduction | 1048576 | 64 | 0.015508 | 0.014476 | 1.07× | -6.7% | improvement | no |
| reduction | 1048576 | 128 | 0.010566 | 0.009525 | 1.11× | -9.8% | improvement | no |
| reduction | 1048576 | 256 | 0.009516 | 0.008616 | 1.10× | -9.5% | improvement | no |
| reduction | 1048576 | 512 | 0.010211 | 0.009178 | 1.11× | -10.1% | improvement | yes |
| reduction | 4194304 | 64 | 0.044893 | 0.044155 | 1.02× | -1.6% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025361 | 0.024354 | 1.04× | -4.0% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021956 | 0.021364 | 1.03× | -2.7% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024113 | 0.022998 | 1.05× | -4.6% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163176 | 0.162397 | 1.00× | -0.5% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084567 | 0.083976 | 1.01× | -0.7% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081276 | 0.080588 | 1.01× | -0.8% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091897 | 0.090646 | 1.01× | -1.4% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636355 | 0.634935 | 1.00× | -0.2% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321434 | 0.320898 | 1.00× | -0.2% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307905 | 0.306423 | 1.00× | -0.5% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350072 | 0.346825 | 1.01× | -0.9% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014884 | 0.013757 | 1.08× | -7.6% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009928 | 0.008913 | 1.11× | -10.2% | improvement | no |
| stencil_1d | 1048576 | 256 | 0.007521 | 0.006333 | 1.19× | -15.8% | improvement | no |
| stencil_1d | 1048576 | 512 | 0.007040 | 0.005823 | 1.21× | -17.3% | improvement | no |
| stencil_1d | 4194304 | 64 | 0.044444 | 0.043448 | 1.02× | -2.2% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024746 | 0.023942 | 1.03× | -3.2% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015430 | 0.014038 | 1.10× | -9.0% | improvement | no |
| stencil_1d | 4194304 | 512 | 0.015503 | 0.012536 | 1.24× | -19.1% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162888 | 0.162180 | 1.00× | -0.4% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084451 | 0.083581 | 1.01× | -1.0% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.064961 | 0.061414 | 1.06× | -5.5% | improvement | no |
| stencil_1d | 16777216 | 512 | 0.068983 | 0.065292 | 1.06× | -5.4% | improvement | no |
| stencil_1d | 67108864 | 64 | 0.635424 | 0.634743 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321469 | 0.320671 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243899 | 0.231048 | 1.06× | -5.3% | improvement | no |
| stencil_1d | 67108864 | 512 | 0.260004 | 0.247301 | 1.05× | -4.9% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014838 | 0.013761 | 1.08× | -7.3% | improvement | no |
| vector_add | 1048576 | 128 | 0.009902 | 0.009146 | 1.08× | -7.6% | improvement | yes |
| vector_add | 1048576 | 256 | 0.007553 | 0.006316 | 1.20× | -16.4% | improvement | yes |
| vector_add | 1048576 | 512 | 0.006889 | 0.005892 | 1.17× | -14.5% | improvement | no |
| vector_add | 4194304 | 64 | 0.044611 | 0.044271 | 1.01× | -0.8% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025137 | 0.024456 | 1.03× | -2.7% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023206 | 0.019202 | 1.21× | -17.3% | improvement | no |
| vector_add | 4194304 | 512 | 0.023794 | 0.020127 | 1.18× | -15.4% | improvement | no |
| vector_add | 16777216 | 64 | 0.162655 | 0.162412 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084577 | 0.083837 | 1.01× | -0.9% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.076918 | 0.067726 | 1.14× | -12.0% | improvement | no |
| vector_add | 16777216 | 512 | 0.079101 | 0.070523 | 1.12× | -10.8% | improvement | no |
| vector_add | 67108864 | 64 | 0.635105 | 0.634925 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321772 | 0.320718 | 1.00× | -0.3% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290443 | 0.256479 | 1.13× | -11.7% | improvement | no |
| vector_add | 67108864 | 512 | 0.298436 | 0.268762 | 1.11× | -9.9% | improvement | no |
