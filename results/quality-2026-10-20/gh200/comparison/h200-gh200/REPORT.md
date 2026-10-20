# Benchmark comparison

NVIDIA H200 → NVIDIA GH200 480GB

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 8.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 1 matched scenarios exceed the CV limit (0.1).
- Gate: 8 matched scenarios exceed the slowdown threshold.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008694 | 0.007612 | 1.14× | -12.4% | improvement | no |
| gemm_naive | 256 | 16 | 0.013634 | 0.012572 | 1.08× | -7.8% | improvement | no |
| gemm_naive | 512 | 16 | 0.060617 | 0.059676 | 1.02× | -1.6% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.418902 | 0.417877 | 1.00× | -0.2% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.178079 | 3.178650 | 1.00× | +0.0% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007561 | 0.006584 | 1.15× | -12.9% | improvement | no |
| gemm_tiled | 256 | 16 | 0.011595 | 0.010625 | 1.09× | -8.4% | improvement | no |
| gemm_tiled | 512 | 16 | 0.042239 | 0.041254 | 1.02× | -2.3% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279226 | 0.278119 | 1.00× | -0.4% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.106378 | 2.106262 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014842 | 0.013776 | 1.08× | -7.2% | improvement | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009973 | 0.008982 | 1.11× | -9.9% | improvement | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007538 | 0.006406 | 1.18× | -15.0% | improvement | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006582 | 0.005477 | 1.20× | -16.8% | improvement | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044479 | 0.043238 | 1.03× | -2.8% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024771 | 0.023921 | 1.04× | -3.4% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015134 | 0.014109 | 1.07× | -6.8% | improvement | no |
| memcpy_bandwidth | 4194304 | 512 | 0.012598 | 0.011203 | 1.12× | -11.1% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162600 | 0.161871 | 1.00× | -0.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084113 | 0.083812 | 1.00× | -0.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.054241 | 0.057028 | 0.95× | +5.1% | regression | no |
| memcpy_bandwidth | 16777216 | 512 | 0.058279 | 0.060775 | 0.96× | +4.3% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 64 | 0.634971 | 0.634477 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320955 | 0.320382 | 1.00× | -0.2% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.201094 | 0.213290 | 0.94× | +6.1% | regression | no |
| memcpy_bandwidth | 67108864 | 512 | 0.217239 | 0.229654 | 0.95× | +5.7% | regression | no |
| reduction | 1048576 | 64 | 0.015482 | 0.014439 | 1.07× | -6.7% | improvement | no |
| reduction | 1048576 | 128 | 0.010612 | 0.009721 | 1.09× | -8.4% | improvement | no |
| reduction | 1048576 | 256 | 0.009644 | 0.008523 | 1.13× | -11.6% | improvement | no |
| reduction | 1048576 | 512 | 0.010042 | 0.009130 | 1.10× | -9.1% | improvement | no |
| reduction | 4194304 | 64 | 0.044819 | 0.044166 | 1.01× | -1.5% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025306 | 0.024408 | 1.04× | -3.6% | within_threshold | no |
| reduction | 4194304 | 256 | 0.022034 | 0.021322 | 1.03× | -3.2% | within_threshold | no |
| reduction | 4194304 | 512 | 0.023987 | 0.023094 | 1.04× | -3.7% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163157 | 0.162306 | 1.01× | -0.5% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084747 | 0.083784 | 1.01× | -1.1% | within_threshold | no |
| reduction | 16777216 | 256 | 0.080777 | 0.080599 | 1.00× | -0.2% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091473 | 0.090620 | 1.01× | -0.9% | within_threshold | no |
| reduction | 67108864 | 64 | 0.635468 | 0.634650 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321361 | 0.320497 | 1.00× | -0.3% | within_threshold | no |
| reduction | 67108864 | 256 | 0.305913 | 0.306657 | 1.00× | +0.2% | within_threshold | no |
| reduction | 67108864 | 512 | 0.347927 | 0.347229 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014966 | 0.013859 | 1.08× | -7.4% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009804 | 0.008922 | 1.10× | -9.0% | improvement | no |
| stencil_1d | 1048576 | 256 | 0.007581 | 0.006350 | 1.19× | -16.2% | improvement | no |
| stencil_1d | 1048576 | 512 | 0.007058 | 0.005844 | 1.21× | -17.2% | improvement | no |
| stencil_1d | 4194304 | 64 | 0.044439 | 0.043846 | 1.01× | -1.3% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024613 | 0.023997 | 1.03× | -2.5% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015019 | 0.014004 | 1.07× | -6.8% | improvement | no |
| stencil_1d | 4194304 | 512 | 0.013874 | 0.012667 | 1.10× | -8.7% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162577 | 0.161935 | 1.00× | -0.4% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084388 | 0.083852 | 1.01× | -0.6% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.059060 | 0.061081 | 0.97× | +3.4% | within_threshold | no |
| stencil_1d | 16777216 | 512 | 0.063128 | 0.065453 | 0.96× | +3.7% | within_threshold | no |
| stencil_1d | 67108864 | 64 | 0.635112 | 0.634183 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321555 | 0.320676 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.220111 | 0.231119 | 0.95× | +5.0% | regression | no |
| stencil_1d | 67108864 | 512 | 0.236063 | 0.247265 | 0.95× | +4.7% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014817 | 0.014006 | 1.06× | -5.5% | improvement | no |
| vector_add | 1048576 | 128 | 0.009949 | 0.008940 | 1.11× | -10.1% | improvement | no |
| vector_add | 1048576 | 256 | 0.007450 | 0.006671 | 1.12× | -10.5% | improvement | yes |
| vector_add | 1048576 | 512 | 0.006885 | 0.005847 | 1.18× | -15.1% | improvement | no |
| vector_add | 4194304 | 64 | 0.044614 | 0.044029 | 1.01× | -1.3% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025012 | 0.024602 | 1.02× | -1.6% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.019358 | 0.018886 | 1.02× | -2.4% | within_threshold | no |
| vector_add | 4194304 | 512 | 0.020317 | 0.020135 | 1.01× | -0.9% | within_threshold | no |
| vector_add | 16777216 | 64 | 0.162736 | 0.162424 | 1.00× | -0.2% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084443 | 0.083850 | 1.01× | -0.7% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.063085 | 0.067420 | 0.94× | +6.9% | regression | no |
| vector_add | 16777216 | 512 | 0.066663 | 0.070457 | 0.95× | +5.7% | regression | no |
| vector_add | 67108864 | 64 | 0.635033 | 0.634674 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321564 | 0.320991 | 1.00× | -0.2% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.236526 | 0.256807 | 0.92× | +8.6% | regression | no |
| vector_add | 67108864 | 512 | 0.250900 | 0.269256 | 0.93× | +7.3% | regression | no |
