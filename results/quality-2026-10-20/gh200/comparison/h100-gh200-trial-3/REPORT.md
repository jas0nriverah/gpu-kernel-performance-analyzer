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
| gemm_naive | 128 | 16 | 0.008785 | 0.007883 | 1.11× | -10.3% | improvement | yes |
| gemm_naive | 256 | 16 | 0.013603 | 0.012456 | 1.09× | -8.4% | improvement | no |
| gemm_naive | 512 | 16 | 0.060324 | 0.059377 | 1.02× | -1.6% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.417904 | 0.418013 | 1.00× | +0.0% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.176542 | 3.179845 | 1.00× | +0.1% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007755 | 0.006406 | 1.21× | -17.4% | improvement | no |
| gemm_tiled | 256 | 16 | 0.011613 | 0.010721 | 1.08× | -7.7% | improvement | no |
| gemm_tiled | 512 | 16 | 0.042124 | 0.041354 | 1.02× | -1.8% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279002 | 0.278275 | 1.00× | -0.3% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.105971 | 2.106173 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014771 | 0.013915 | 1.06× | -5.8% | improvement | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009917 | 0.008835 | 1.12× | -10.9% | improvement | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007516 | 0.006424 | 1.17× | -14.5% | improvement | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006873 | 0.005728 | 1.20× | -16.7% | improvement | yes |
| memcpy_bandwidth | 4194304 | 64 | 0.044410 | 0.043624 | 1.02× | -1.8% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024728 | 0.024093 | 1.03× | -2.6% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015114 | 0.014261 | 1.06× | -5.6% | improvement | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013636 | 0.011252 | 1.21× | -17.5% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162687 | 0.161847 | 1.01× | -0.5% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084345 | 0.083624 | 1.01× | -0.9% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060518 | 0.056946 | 1.06× | -5.9% | improvement | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064813 | 0.061068 | 1.06× | -5.8% | improvement | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635429 | 0.634258 | 1.00× | -0.2% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320965 | 0.320090 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225791 | 0.213354 | 1.06× | -5.5% | improvement | no |
| memcpy_bandwidth | 67108864 | 512 | 0.243247 | 0.229526 | 1.06× | -5.6% | improvement | no |
| reduction | 1048576 | 64 | 0.015519 | 0.014505 | 1.07× | -6.5% | improvement | no |
| reduction | 1048576 | 128 | 0.010466 | 0.009610 | 1.09× | -8.2% | improvement | no |
| reduction | 1048576 | 256 | 0.009472 | 0.008754 | 1.08× | -7.6% | improvement | yes |
| reduction | 1048576 | 512 | 0.010090 | 0.009100 | 1.11× | -9.8% | improvement | no |
| reduction | 4194304 | 64 | 0.045012 | 0.043981 | 1.02× | -2.3% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025366 | 0.024420 | 1.04× | -3.7% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021939 | 0.021245 | 1.03× | -3.2% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024052 | 0.023289 | 1.03× | -3.2% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163320 | 0.162690 | 1.00× | -0.4% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084659 | 0.083990 | 1.01× | -0.8% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081378 | 0.080706 | 1.01× | -0.8% | within_threshold | no |
| reduction | 16777216 | 512 | 0.092095 | 0.090657 | 1.02× | -1.6% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636219 | 0.634771 | 1.00× | -0.2% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321443 | 0.320742 | 1.00× | -0.2% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307786 | 0.306877 | 1.00× | -0.3% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350141 | 0.347149 | 1.01× | -0.9% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014820 | 0.013818 | 1.07× | -6.8% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009889 | 0.008915 | 1.11× | -9.8% | improvement | no |
| stencil_1d | 1048576 | 256 | 0.007560 | 0.006433 | 1.18× | -14.9% | improvement | no |
| stencil_1d | 1048576 | 512 | 0.007032 | 0.005860 | 1.20× | -16.7% | improvement | no |
| stencil_1d | 4194304 | 64 | 0.044494 | 0.043685 | 1.02× | -1.8% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024915 | 0.023942 | 1.04× | -3.9% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015452 | 0.014127 | 1.09× | -8.6% | improvement | no |
| stencil_1d | 4194304 | 512 | 0.015644 | 0.012673 | 1.23× | -19.0% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162813 | 0.162115 | 1.00× | -0.4% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084360 | 0.083515 | 1.01× | -1.0% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.065264 | 0.061375 | 1.06× | -6.0% | improvement | no |
| stencil_1d | 16777216 | 512 | 0.069178 | 0.065404 | 1.06× | -5.5% | improvement | no |
| stencil_1d | 67108864 | 64 | 0.635449 | 0.634126 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321286 | 0.320542 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243771 | 0.231284 | 1.05× | -5.1% | improvement | no |
| stencil_1d | 67108864 | 512 | 0.259863 | 0.247148 | 1.05× | -4.9% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014826 | 0.013837 | 1.07× | -6.7% | improvement | no |
| vector_add | 1048576 | 128 | 0.009852 | 0.008925 | 1.10× | -9.4% | improvement | no |
| vector_add | 1048576 | 256 | 0.007508 | 0.006624 | 1.13× | -11.8% | improvement | no |
| vector_add | 1048576 | 512 | 0.006865 | 0.005739 | 1.20× | -16.4% | improvement | no |
| vector_add | 4194304 | 64 | 0.044484 | 0.043953 | 1.01× | -1.2% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025130 | 0.024411 | 1.03× | -2.9% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023133 | 0.019315 | 1.20× | -16.5% | improvement | no |
| vector_add | 4194304 | 512 | 0.023780 | 0.020073 | 1.18× | -15.6% | improvement | no |
| vector_add | 16777216 | 64 | 0.162659 | 0.162109 | 1.00× | -0.3% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084404 | 0.083906 | 1.01× | -0.6% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.077044 | 0.067798 | 1.14× | -12.0% | improvement | no |
| vector_add | 16777216 | 512 | 0.079151 | 0.070421 | 1.12× | -11.0% | improvement | no |
| vector_add | 67108864 | 64 | 0.634912 | 0.634807 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321834 | 0.320976 | 1.00× | -0.3% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290708 | 0.256901 | 1.13× | -11.6% | improvement | no |
| vector_add | 67108864 | 512 | 0.298653 | 0.269025 | 1.11× | -9.9% | improvement | no |
