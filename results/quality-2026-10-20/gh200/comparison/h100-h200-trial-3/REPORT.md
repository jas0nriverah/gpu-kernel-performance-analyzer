# Benchmark comparison

NVIDIA H100 80GB HBM3 → NVIDIA H200

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
| gemm_naive | 128 | 16 | 0.008785 | 0.008535 | 1.03× | -2.8% | within_threshold | no |
| gemm_naive | 256 | 16 | 0.013603 | 0.013607 | 1.00× | +0.0% | within_threshold | no |
| gemm_naive | 512 | 16 | 0.060324 | 0.060509 | 1.00× | +0.3% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.417904 | 0.418365 | 1.00× | +0.1% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.176542 | 3.177724 | 1.00× | +0.0% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007755 | 0.007400 | 1.05× | -4.6% | within_threshold | no |
| gemm_tiled | 256 | 16 | 0.011613 | 0.011638 | 1.00× | +0.2% | within_threshold | no |
| gemm_tiled | 512 | 16 | 0.042124 | 0.042098 | 1.00× | -0.1% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279002 | 0.279204 | 1.00× | +0.1% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.105971 | 2.106435 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014771 | 0.014849 | 0.99× | +0.5% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009917 | 0.009735 | 1.02× | -1.8% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007516 | 0.007283 | 1.03× | -3.1% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006873 | 0.006660 | 1.03× | -3.1% | within_threshold | yes |
| memcpy_bandwidth | 4194304 | 64 | 0.044410 | 0.044270 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024728 | 0.024650 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015114 | 0.014870 | 1.02× | -1.6% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013636 | 0.012328 | 1.11× | -9.6% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162687 | 0.162387 | 1.00× | -0.2% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084345 | 0.084024 | 1.00× | -0.4% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060518 | 0.054234 | 1.12× | -10.4% | improvement | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064813 | 0.058014 | 1.12× | -10.5% | improvement | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635429 | 0.634991 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320965 | 0.320925 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225791 | 0.201066 | 1.12× | -11.0% | improvement | no |
| memcpy_bandwidth | 67108864 | 512 | 0.243247 | 0.217326 | 1.12× | -10.7% | improvement | no |
| reduction | 1048576 | 64 | 0.015519 | 0.015269 | 1.02× | -1.6% | within_threshold | no |
| reduction | 1048576 | 128 | 0.010466 | 0.010479 | 1.00× | +0.1% | within_threshold | no |
| reduction | 1048576 | 256 | 0.009472 | 0.009504 | 1.00× | +0.3% | within_threshold | no |
| reduction | 1048576 | 512 | 0.010090 | 0.009951 | 1.01× | -1.4% | within_threshold | no |
| reduction | 4194304 | 64 | 0.045012 | 0.044942 | 1.00× | -0.2% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025366 | 0.025322 | 1.00× | -0.2% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021939 | 0.021955 | 1.00× | +0.1% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024052 | 0.023862 | 1.01× | -0.8% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163320 | 0.163048 | 1.00× | -0.2% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084659 | 0.084775 | 1.00× | +0.1% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081378 | 0.080980 | 1.00× | -0.5% | within_threshold | no |
| reduction | 16777216 | 512 | 0.092095 | 0.091471 | 1.01× | -0.7% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636219 | 0.635512 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321443 | 0.321372 | 1.00× | -0.0% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307786 | 0.305980 | 1.01× | -0.6% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350141 | 0.347863 | 1.01× | -0.7% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014820 | 0.014855 | 1.00× | +0.2% | within_threshold | no |
| stencil_1d | 1048576 | 128 | 0.009889 | 0.009919 | 1.00× | +0.3% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007560 | 0.007604 | 0.99× | +0.6% | within_threshold | no |
| stencil_1d | 1048576 | 512 | 0.007032 | 0.006993 | 1.01× | -0.6% | within_threshold | no |
| stencil_1d | 4194304 | 64 | 0.044494 | 0.044351 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024915 | 0.024710 | 1.01× | -0.8% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015452 | 0.015071 | 1.03× | -2.5% | within_threshold | no |
| stencil_1d | 4194304 | 512 | 0.015644 | 0.013756 | 1.14× | -12.1% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162813 | 0.162594 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084360 | 0.084264 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.065264 | 0.058954 | 1.11× | -9.7% | improvement | no |
| stencil_1d | 16777216 | 512 | 0.069178 | 0.063097 | 1.10× | -8.8% | improvement | no |
| stencil_1d | 67108864 | 64 | 0.635449 | 0.635005 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321286 | 0.321391 | 1.00× | +0.0% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243771 | 0.219999 | 1.11× | -9.8% | improvement | no |
| stencil_1d | 67108864 | 512 | 0.259863 | 0.236125 | 1.10× | -9.1% | improvement | no |
| vector_add | 1048576 | 64 | 0.014826 | 0.014818 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 1048576 | 128 | 0.009852 | 0.009742 | 1.01× | -1.1% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007508 | 0.007460 | 1.01× | -0.6% | within_threshold | no |
| vector_add | 1048576 | 512 | 0.006865 | 0.006682 | 1.03× | -2.7% | within_threshold | no |
| vector_add | 4194304 | 64 | 0.044484 | 0.044570 | 1.00× | +0.2% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025130 | 0.025100 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023133 | 0.019192 | 1.21× | -17.0% | improvement | no |
| vector_add | 4194304 | 512 | 0.023780 | 0.020295 | 1.17× | -14.7% | improvement | no |
| vector_add | 16777216 | 64 | 0.162659 | 0.162735 | 1.00× | +0.0% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084404 | 0.084349 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.077044 | 0.062930 | 1.22× | -18.3% | improvement | no |
| vector_add | 16777216 | 512 | 0.079151 | 0.066664 | 1.19× | -15.8% | improvement | no |
| vector_add | 67108864 | 64 | 0.634912 | 0.634989 | 1.00× | +0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321834 | 0.321458 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290708 | 0.236369 | 1.23× | -18.7% | improvement | no |
| vector_add | 67108864 | 512 | 0.298653 | 0.250720 | 1.19× | -16.0% | improvement | no |
