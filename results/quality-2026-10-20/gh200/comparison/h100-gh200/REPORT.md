# Benchmark comparison

NVIDIA H100 80GB HBM3 → NVIDIA GH200 480GB

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 0.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 1 matched scenarios exceed the CV limit (0.1).
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: git_commit.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.
- Run metadata differs: source_sha256.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008832 | 0.007612 | 1.16× | -13.8% | improvement | no |
| gemm_naive | 256 | 16 | 0.013674 | 0.012572 | 1.09× | -8.1% | improvement | no |
| gemm_naive | 512 | 16 | 0.060282 | 0.059676 | 1.01× | -1.0% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.418256 | 0.417877 | 1.00× | -0.1% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.159608 | 3.178650 | 0.99× | +0.6% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007690 | 0.006584 | 1.17× | -14.4% | improvement | no |
| gemm_tiled | 256 | 16 | 0.011566 | 0.010625 | 1.09× | -8.1% | improvement | no |
| gemm_tiled | 512 | 16 | 0.041894 | 0.041254 | 1.02× | -1.5% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.276892 | 0.278119 | 1.00× | +0.4% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.090242 | 2.106262 | 0.99× | +0.8% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014817 | 0.013776 | 1.08× | -7.0% | improvement | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009880 | 0.008982 | 1.10× | -9.1% | improvement | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007391 | 0.006406 | 1.15× | -13.3% | improvement | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006709 | 0.005477 | 1.22× | -18.4% | improvement | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044471 | 0.043238 | 1.03× | -2.8% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024703 | 0.023921 | 1.03× | -3.2% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015071 | 0.014109 | 1.07× | -6.4% | improvement | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013736 | 0.011203 | 1.23× | -18.4% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162840 | 0.161871 | 1.01× | -0.6% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084141 | 0.083812 | 1.00× | -0.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060476 | 0.057028 | 1.06× | -5.7% | improvement | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064798 | 0.060775 | 1.07× | -6.2% | improvement | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635199 | 0.634477 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320904 | 0.320382 | 1.00× | -0.2% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225754 | 0.213290 | 1.06× | -5.5% | improvement | no |
| memcpy_bandwidth | 67108864 | 512 | 0.243235 | 0.229654 | 1.06× | -5.6% | improvement | no |
| reduction | 1048576 | 64 | 0.015467 | 0.014439 | 1.07× | -6.6% | improvement | no |
| reduction | 1048576 | 128 | 0.010579 | 0.009721 | 1.09× | -8.1% | improvement | no |
| reduction | 1048576 | 256 | 0.009557 | 0.008523 | 1.12× | -10.8% | improvement | no |
| reduction | 1048576 | 512 | 0.010067 | 0.009130 | 1.10× | -9.3% | improvement | no |
| reduction | 4194304 | 64 | 0.044980 | 0.044166 | 1.02× | -1.8% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025431 | 0.024408 | 1.04× | -4.0% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021996 | 0.021322 | 1.03× | -3.1% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024077 | 0.023094 | 1.04× | -4.1% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163196 | 0.162306 | 1.01× | -0.5% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084748 | 0.083784 | 1.01× | -1.1% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081283 | 0.080599 | 1.01× | -0.8% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091993 | 0.090620 | 1.02× | -1.5% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636312 | 0.634650 | 1.00× | -0.3% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321516 | 0.320497 | 1.00× | -0.3% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307970 | 0.306657 | 1.00× | -0.4% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350140 | 0.347229 | 1.01× | -0.8% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014930 | 0.013859 | 1.08× | -7.2% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009933 | 0.008922 | 1.11× | -10.2% | improvement | no |
| stencil_1d | 1048576 | 256 | 0.007495 | 0.006350 | 1.18× | -15.3% | improvement | no |
| stencil_1d | 1048576 | 512 | 0.007018 | 0.005844 | 1.20× | -16.7% | improvement | no |
| stencil_1d | 4194304 | 64 | 0.044474 | 0.043846 | 1.01× | -1.4% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.025047 | 0.023997 | 1.04× | -4.2% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015590 | 0.014004 | 1.11× | -10.2% | improvement | no |
| stencil_1d | 4194304 | 512 | 0.015658 | 0.012667 | 1.24× | -19.1% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162772 | 0.161935 | 1.01× | -0.5% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084416 | 0.083852 | 1.01× | -0.7% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.065088 | 0.061081 | 1.07× | -6.2% | improvement | no |
| stencil_1d | 16777216 | 512 | 0.069218 | 0.065453 | 1.06× | -5.4% | improvement | no |
| stencil_1d | 67108864 | 64 | 0.635241 | 0.634183 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321335 | 0.320676 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243750 | 0.231119 | 1.05× | -5.2% | improvement | no |
| stencil_1d | 67108864 | 512 | 0.259904 | 0.247265 | 1.05× | -4.9% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014799 | 0.014006 | 1.06× | -5.4% | improvement | no |
| vector_add | 1048576 | 128 | 0.009915 | 0.008940 | 1.11× | -9.8% | improvement | no |
| vector_add | 1048576 | 256 | 0.007434 | 0.006671 | 1.11× | -10.3% | improvement | yes |
| vector_add | 1048576 | 512 | 0.006800 | 0.005847 | 1.16× | -14.0% | improvement | no |
| vector_add | 4194304 | 64 | 0.044617 | 0.044029 | 1.01× | -1.3% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025122 | 0.024602 | 1.02× | -2.1% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023203 | 0.018886 | 1.23× | -18.6% | improvement | no |
| vector_add | 4194304 | 512 | 0.023811 | 0.020135 | 1.18× | -15.4% | improvement | no |
| vector_add | 16777216 | 64 | 0.162646 | 0.162424 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084497 | 0.083850 | 1.01× | -0.8% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.076991 | 0.067420 | 1.14× | -12.4% | improvement | no |
| vector_add | 16777216 | 512 | 0.079134 | 0.070457 | 1.12× | -11.0% | improvement | no |
| vector_add | 67108864 | 64 | 0.635240 | 0.634674 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321926 | 0.320991 | 1.00× | -0.3% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290634 | 0.256807 | 1.13× | -11.6% | improvement | no |
| vector_add | 67108864 | 512 | 0.298547 | 0.269256 | 1.11× | -9.8% | improvement | no |
