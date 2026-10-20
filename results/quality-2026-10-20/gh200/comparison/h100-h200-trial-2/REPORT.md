# Benchmark comparison

NVIDIA H100 80GB HBM3 → NVIDIA H200

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 0.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 2 matched scenarios exceed the CV limit (0.1).
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: git_commit.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.
- Run metadata differs: source_sha256.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008813 | 0.008548 | 1.03× | -3.0% | within_threshold | no |
| gemm_naive | 256 | 16 | 0.013612 | 0.013544 | 1.01× | -0.5% | within_threshold | no |
| gemm_naive | 512 | 16 | 0.060358 | 0.060570 | 1.00× | +0.4% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.417903 | 0.418335 | 1.00× | +0.1% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.177004 | 3.178168 | 1.00× | +0.0% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007628 | 0.007583 | 1.01× | -0.6% | within_threshold | yes |
| gemm_tiled | 256 | 16 | 0.011645 | 0.011635 | 1.00× | -0.1% | within_threshold | no |
| gemm_tiled | 512 | 16 | 0.042056 | 0.042013 | 1.00× | -0.1% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279035 | 0.279112 | 1.00× | +0.0% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.105848 | 2.106191 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014764 | 0.014721 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009858 | 0.009916 | 0.99× | +0.6% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007512 | 0.007322 | 1.03× | -2.5% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006662 | 0.006496 | 1.03× | -2.5% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044412 | 0.044360 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024829 | 0.024550 | 1.01× | -1.1% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015075 | 0.014889 | 1.01× | -1.2% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013793 | 0.012331 | 1.12× | -10.6% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162644 | 0.162598 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084159 | 0.084113 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060468 | 0.054352 | 1.11× | -10.1% | improvement | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064780 | 0.058319 | 1.11× | -10.0% | improvement | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635264 | 0.635093 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320990 | 0.321023 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225416 | 0.201139 | 1.12× | -10.8% | improvement | no |
| memcpy_bandwidth | 67108864 | 512 | 0.242744 | 0.217191 | 1.12× | -10.5% | improvement | no |
| reduction | 1048576 | 64 | 0.015508 | 0.015455 | 1.00× | -0.3% | within_threshold | no |
| reduction | 1048576 | 128 | 0.010566 | 0.010454 | 1.01× | -1.1% | within_threshold | no |
| reduction | 1048576 | 256 | 0.009516 | 0.009491 | 1.00× | -0.3% | within_threshold | no |
| reduction | 1048576 | 512 | 0.010211 | 0.010005 | 1.02× | -2.0% | within_threshold | no |
| reduction | 4194304 | 64 | 0.044893 | 0.044938 | 1.00× | +0.1% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025361 | 0.025323 | 1.00× | -0.2% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021956 | 0.022025 | 1.00× | +0.3% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024113 | 0.024013 | 1.00× | -0.4% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163176 | 0.163028 | 1.00× | -0.1% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084567 | 0.084725 | 1.00× | +0.2% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081276 | 0.080714 | 1.01× | -0.7% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091897 | 0.091401 | 1.01× | -0.5% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636355 | 0.635498 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321434 | 0.321309 | 1.00× | -0.0% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307905 | 0.305892 | 1.01× | -0.7% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350072 | 0.347774 | 1.01× | -0.7% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014884 | 0.014896 | 1.00× | +0.1% | within_threshold | no |
| stencil_1d | 1048576 | 128 | 0.009928 | 0.009918 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007521 | 0.007478 | 1.01× | -0.6% | within_threshold | no |
| stencil_1d | 1048576 | 512 | 0.007040 | 0.006979 | 1.01× | -0.9% | within_threshold | no |
| stencil_1d | 4194304 | 64 | 0.044444 | 0.044415 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024746 | 0.024765 | 1.00× | +0.1% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015430 | 0.015068 | 1.02× | -2.3% | within_threshold | no |
| stencil_1d | 4194304 | 512 | 0.015503 | 0.013769 | 1.13× | -11.2% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162888 | 0.162627 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084451 | 0.084244 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.064961 | 0.059092 | 1.10× | -9.0% | improvement | no |
| stencil_1d | 16777216 | 512 | 0.068983 | 0.063089 | 1.09× | -8.5% | improvement | no |
| stencil_1d | 67108864 | 64 | 0.635424 | 0.635024 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321469 | 0.321463 | 1.00× | -0.0% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243899 | 0.220058 | 1.11× | -9.8% | improvement | no |
| stencil_1d | 67108864 | 512 | 0.260004 | 0.236023 | 1.10× | -9.2% | improvement | no |
| vector_add | 1048576 | 64 | 0.014838 | 0.014755 | 1.01× | -0.6% | within_threshold | no |
| vector_add | 1048576 | 128 | 0.009902 | 0.009835 | 1.01× | -0.7% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007553 | 0.007455 | 1.01× | -1.3% | within_threshold | yes |
| vector_add | 1048576 | 512 | 0.006889 | 0.006644 | 1.04× | -3.5% | within_threshold | no |
| vector_add | 4194304 | 64 | 0.044611 | 0.044505 | 1.00× | -0.2% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025137 | 0.025014 | 1.00× | -0.5% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023206 | 0.019414 | 1.20× | -16.3% | improvement | no |
| vector_add | 4194304 | 512 | 0.023794 | 0.020371 | 1.17× | -14.4% | improvement | no |
| vector_add | 16777216 | 64 | 0.162655 | 0.162869 | 1.00× | +0.1% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084577 | 0.084440 | 1.00× | -0.2% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.076918 | 0.063150 | 1.22× | -17.9% | improvement | no |
| vector_add | 16777216 | 512 | 0.079101 | 0.066689 | 1.19× | -15.7% | improvement | no |
| vector_add | 67108864 | 64 | 0.635105 | 0.635110 | 1.00× | +0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321772 | 0.321651 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290443 | 0.236536 | 1.23× | -18.6% | improvement | no |
| vector_add | 67108864 | 512 | 0.298436 | 0.250788 | 1.19× | -16.0% | improvement | no |
