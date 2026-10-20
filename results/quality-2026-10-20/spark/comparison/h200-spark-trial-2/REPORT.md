# Benchmark comparison

NVIDIA H200 → NVIDIA GB10

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 66.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 3 matched scenarios exceed the CV limit (0.1).
- Gate: 66 matched scenarios exceed the slowdown threshold.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008548 | 0.007569 | 1.13× | -11.5% | improvement | no |
| gemm_naive | 256 | 16 | 0.013544 | 0.027679 | 0.49× | +104.4% | regression | no |
| gemm_naive | 512 | 16 | 0.060570 | 0.153524 | 0.39× | +153.5% | regression | no |
| gemm_naive | 1024 | 16 | 0.418335 | 1.152179 | 0.36× | +175.4% | regression | no |
| gemm_naive | 2048 | 16 | 3.178168 | 9.149414 | 0.35× | +187.9% | regression | no |
| gemm_tiled | 128 | 16 | 0.007583 | 0.007574 | 1.00× | -0.1% | within_threshold | yes |
| gemm_tiled | 256 | 16 | 0.011635 | 0.020601 | 0.56× | +77.1% | regression | no |
| gemm_tiled | 512 | 16 | 0.042013 | 0.115586 | 0.36× | +175.1% | regression | no |
| gemm_tiled | 1024 | 16 | 0.279112 | 0.872060 | 0.32× | +212.4% | regression | no |
| gemm_tiled | 2048 | 16 | 2.106191 | 6.860309 | 0.31× | +225.7% | regression | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014721 | 0.014968 | 0.98× | +1.7% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009916 | 0.009848 | 1.01× | -0.7% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007322 | 0.010199 | 0.72× | +39.3% | regression | yes |
| memcpy_bandwidth | 1048576 | 512 | 0.006496 | 0.009912 | 0.66× | +52.6% | regression | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044360 | 0.115462 | 0.38× | +160.3% | regression | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024550 | 0.125879 | 0.20× | +412.8% | regression | no |
| memcpy_bandwidth | 4194304 | 256 | 0.014889 | 0.124680 | 0.12× | +737.4% | regression | no |
| memcpy_bandwidth | 4194304 | 512 | 0.012331 | 0.114614 | 0.11× | +829.5% | regression | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162598 | 0.557101 | 0.29× | +242.6% | regression | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084113 | 0.545517 | 0.15× | +548.6% | regression | no |
| memcpy_bandwidth | 16777216 | 256 | 0.054352 | 0.543643 | 0.10× | +900.2% | regression | no |
| memcpy_bandwidth | 16777216 | 512 | 0.058319 | 0.544277 | 0.11× | +833.3% | regression | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635093 | 2.174241 | 0.29× | +242.3% | regression | no |
| memcpy_bandwidth | 67108864 | 128 | 0.321023 | 2.190303 | 0.15× | +582.3% | regression | no |
| memcpy_bandwidth | 67108864 | 256 | 0.201139 | 2.174147 | 0.09× | +980.9% | regression | no |
| memcpy_bandwidth | 67108864 | 512 | 0.217191 | 2.186477 | 0.10× | +906.7% | regression | no |
| reduction | 1048576 | 64 | 0.015455 | 0.019640 | 0.79× | +27.1% | regression | no |
| reduction | 1048576 | 128 | 0.010454 | 0.019679 | 0.53× | +88.2% | regression | no |
| reduction | 1048576 | 256 | 0.009491 | 0.020028 | 0.47× | +111.0% | regression | no |
| reduction | 1048576 | 512 | 0.010005 | 0.022051 | 0.45× | +120.4% | regression | no |
| reduction | 4194304 | 64 | 0.044938 | 0.062584 | 0.72× | +39.3% | regression | no |
| reduction | 4194304 | 128 | 0.025323 | 0.062918 | 0.40× | +148.5% | regression | no |
| reduction | 4194304 | 256 | 0.022025 | 0.066840 | 0.33× | +203.5% | regression | no |
| reduction | 4194304 | 512 | 0.024013 | 0.074969 | 0.32× | +212.2% | regression | no |
| reduction | 16777216 | 64 | 0.163028 | 0.264916 | 0.62× | +62.5% | regression | no |
| reduction | 16777216 | 128 | 0.084725 | 0.268679 | 0.32× | +217.1% | regression | no |
| reduction | 16777216 | 256 | 0.080714 | 0.291753 | 0.28× | +261.5% | regression | no |
| reduction | 16777216 | 512 | 0.091401 | 0.339000 | 0.27× | +270.9% | regression | no |
| reduction | 67108864 | 64 | 0.635498 | 1.085294 | 0.59× | +70.8% | regression | no |
| reduction | 67108864 | 128 | 0.321309 | 1.103442 | 0.29× | +243.4% | regression | no |
| reduction | 67108864 | 256 | 0.305892 | 1.203874 | 0.25× | +293.6% | regression | no |
| reduction | 67108864 | 512 | 0.347774 | 1.390434 | 0.25× | +299.8% | regression | no |
| stencil_1d | 1048576 | 64 | 0.014896 | 0.014139 | 1.05× | -5.1% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009918 | 0.009642 | 1.03× | -2.8% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007478 | 0.009693 | 0.77× | +29.6% | regression | no |
| stencil_1d | 1048576 | 512 | 0.006979 | 0.009967 | 0.70× | +42.8% | regression | yes |
| stencil_1d | 4194304 | 64 | 0.044415 | 0.116012 | 0.38× | +161.2% | regression | no |
| stencil_1d | 4194304 | 128 | 0.024765 | 0.132584 | 0.19× | +435.4% | regression | no |
| stencil_1d | 4194304 | 256 | 0.015068 | 0.130718 | 0.12× | +767.5% | regression | no |
| stencil_1d | 4194304 | 512 | 0.013769 | 0.131532 | 0.10× | +855.3% | regression | no |
| stencil_1d | 16777216 | 64 | 0.162627 | 0.574136 | 0.28× | +253.0% | regression | no |
| stencil_1d | 16777216 | 128 | 0.084244 | 0.558232 | 0.15× | +562.6% | regression | no |
| stencil_1d | 16777216 | 256 | 0.059092 | 0.566284 | 0.10× | +858.3% | regression | no |
| stencil_1d | 16777216 | 512 | 0.063089 | 0.568705 | 0.11× | +801.4% | regression | no |
| stencil_1d | 67108864 | 64 | 0.635024 | 2.315928 | 0.27× | +264.7% | regression | no |
| stencil_1d | 67108864 | 128 | 0.321463 | 2.244602 | 0.14× | +598.2% | regression | no |
| stencil_1d | 67108864 | 256 | 0.220058 | 2.226776 | 0.10× | +911.9% | regression | no |
| stencil_1d | 67108864 | 512 | 0.236023 | 2.263895 | 0.10× | +859.2% | regression | no |
| vector_add | 1048576 | 64 | 0.014755 | 0.013925 | 1.06× | -5.6% | improvement | no |
| vector_add | 1048576 | 128 | 0.009835 | 0.009841 | 1.00× | +0.1% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007455 | 0.009992 | 0.75× | +34.0% | regression | no |
| vector_add | 1048576 | 512 | 0.006644 | 0.009900 | 0.67× | +49.0% | regression | no |
| vector_add | 4194304 | 64 | 0.044505 | 0.195648 | 0.23× | +339.6% | regression | no |
| vector_add | 4194304 | 128 | 0.025014 | 0.198637 | 0.13× | +694.1% | regression | no |
| vector_add | 4194304 | 256 | 0.019414 | 0.200222 | 0.10× | +931.3% | regression | no |
| vector_add | 4194304 | 512 | 0.020371 | 0.188950 | 0.11× | +827.6% | regression | no |
| vector_add | 16777216 | 64 | 0.162869 | 0.816128 | 0.20× | +401.1% | regression | no |
| vector_add | 16777216 | 128 | 0.084440 | 0.813155 | 0.10× | +863.0% | regression | no |
| vector_add | 16777216 | 256 | 0.063150 | 0.815261 | 0.08× | +1191.0% | regression | no |
| vector_add | 16777216 | 512 | 0.066689 | 0.812479 | 0.08× | +1118.3% | regression | no |
| vector_add | 67108864 | 64 | 0.635110 | 3.263859 | 0.19× | +413.9% | regression | no |
| vector_add | 67108864 | 128 | 0.321651 | 3.266532 | 0.10× | +915.6% | regression | no |
| vector_add | 67108864 | 256 | 0.236536 | 3.264695 | 0.07× | +1280.2% | regression | no |
| vector_add | 67108864 | 512 | 0.250788 | 3.252543 | 0.08× | +1196.9% | regression | no |
