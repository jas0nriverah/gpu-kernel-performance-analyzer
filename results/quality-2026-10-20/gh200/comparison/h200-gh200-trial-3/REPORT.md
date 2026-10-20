# Benchmark comparison

NVIDIA H200 → NVIDIA GH200 480GB

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 9.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 3 matched scenarios exceed the CV limit (0.1).
- Gate: 9 matched scenarios exceed the slowdown threshold.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008535 | 0.007883 | 1.08× | -7.6% | improvement | yes |
| gemm_naive | 256 | 16 | 0.013607 | 0.012456 | 1.09× | -8.5% | improvement | no |
| gemm_naive | 512 | 16 | 0.060509 | 0.059377 | 1.02× | -1.9% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.418365 | 0.418013 | 1.00× | -0.1% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.177724 | 3.179845 | 1.00× | +0.1% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007400 | 0.006406 | 1.16× | -13.4% | improvement | no |
| gemm_tiled | 256 | 16 | 0.011638 | 0.010721 | 1.09× | -7.9% | improvement | no |
| gemm_tiled | 512 | 16 | 0.042098 | 0.041354 | 1.02× | -1.8% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.279204 | 0.278275 | 1.00× | -0.3% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.106435 | 2.106173 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014849 | 0.013915 | 1.07× | -6.3% | improvement | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009735 | 0.008835 | 1.10× | -9.2% | improvement | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007283 | 0.006424 | 1.13× | -11.8% | improvement | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006660 | 0.005728 | 1.16× | -14.0% | improvement | yes |
| memcpy_bandwidth | 4194304 | 64 | 0.044270 | 0.043624 | 1.01× | -1.5% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024650 | 0.024093 | 1.02× | -2.3% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.014870 | 0.014261 | 1.04× | -4.1% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 512 | 0.012328 | 0.011252 | 1.10× | -8.7% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162387 | 0.161847 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084024 | 0.083624 | 1.00× | -0.5% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.054234 | 0.056946 | 0.95× | +5.0% | regression | no |
| memcpy_bandwidth | 16777216 | 512 | 0.058014 | 0.061068 | 0.95× | +5.3% | regression | no |
| memcpy_bandwidth | 67108864 | 64 | 0.634991 | 0.634258 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320925 | 0.320090 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.201066 | 0.213354 | 0.94× | +6.1% | regression | no |
| memcpy_bandwidth | 67108864 | 512 | 0.217326 | 0.229526 | 0.95× | +5.6% | regression | no |
| reduction | 1048576 | 64 | 0.015269 | 0.014505 | 1.05× | -5.0% | improvement | no |
| reduction | 1048576 | 128 | 0.010479 | 0.009610 | 1.09× | -8.3% | improvement | no |
| reduction | 1048576 | 256 | 0.009504 | 0.008754 | 1.09× | -7.9% | improvement | yes |
| reduction | 1048576 | 512 | 0.009951 | 0.009100 | 1.09× | -8.6% | improvement | no |
| reduction | 4194304 | 64 | 0.044942 | 0.043981 | 1.02× | -2.1% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025322 | 0.024420 | 1.04× | -3.6% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021955 | 0.021245 | 1.03× | -3.2% | within_threshold | no |
| reduction | 4194304 | 512 | 0.023862 | 0.023289 | 1.02× | -2.4% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163048 | 0.162690 | 1.00× | -0.2% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084775 | 0.083990 | 1.01× | -0.9% | within_threshold | no |
| reduction | 16777216 | 256 | 0.080980 | 0.080706 | 1.00× | -0.3% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091471 | 0.090657 | 1.01× | -0.9% | within_threshold | no |
| reduction | 67108864 | 64 | 0.635512 | 0.634771 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321372 | 0.320742 | 1.00× | -0.2% | within_threshold | no |
| reduction | 67108864 | 256 | 0.305980 | 0.306877 | 1.00× | +0.3% | within_threshold | no |
| reduction | 67108864 | 512 | 0.347863 | 0.347149 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014855 | 0.013818 | 1.08× | -7.0% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009919 | 0.008915 | 1.11× | -10.1% | improvement | no |
| stencil_1d | 1048576 | 256 | 0.007604 | 0.006433 | 1.18× | -15.4% | improvement | no |
| stencil_1d | 1048576 | 512 | 0.006993 | 0.005860 | 1.19× | -16.2% | improvement | no |
| stencil_1d | 4194304 | 64 | 0.044351 | 0.043685 | 1.02× | -1.5% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024710 | 0.023942 | 1.03× | -3.1% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015071 | 0.014127 | 1.07× | -6.3% | improvement | no |
| stencil_1d | 4194304 | 512 | 0.013756 | 0.012673 | 1.09× | -7.9% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162594 | 0.162115 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084264 | 0.083515 | 1.01× | -0.9% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.058954 | 0.061375 | 0.96× | +4.1% | within_threshold | no |
| stencil_1d | 16777216 | 512 | 0.063097 | 0.065404 | 0.96× | +3.7% | within_threshold | no |
| stencil_1d | 67108864 | 64 | 0.635005 | 0.634126 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321391 | 0.320542 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.219999 | 0.231284 | 0.95× | +5.1% | regression | no |
| stencil_1d | 67108864 | 512 | 0.236125 | 0.247148 | 0.96× | +4.7% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014818 | 0.013837 | 1.07× | -6.6% | improvement | no |
| vector_add | 1048576 | 128 | 0.009742 | 0.008925 | 1.09× | -8.4% | improvement | no |
| vector_add | 1048576 | 256 | 0.007460 | 0.006624 | 1.13× | -11.2% | improvement | no |
| vector_add | 1048576 | 512 | 0.006682 | 0.005739 | 1.16× | -14.1% | improvement | no |
| vector_add | 4194304 | 64 | 0.044570 | 0.043953 | 1.01× | -1.4% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025100 | 0.024411 | 1.03× | -2.7% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.019192 | 0.019315 | 0.99× | +0.6% | within_threshold | no |
| vector_add | 4194304 | 512 | 0.020295 | 0.020073 | 1.01× | -1.1% | within_threshold | no |
| vector_add | 16777216 | 64 | 0.162735 | 0.162109 | 1.00× | -0.4% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084349 | 0.083906 | 1.01× | -0.5% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.062930 | 0.067798 | 0.93× | +7.7% | regression | no |
| vector_add | 16777216 | 512 | 0.066664 | 0.070421 | 0.95× | +5.6% | regression | no |
| vector_add | 67108864 | 64 | 0.634989 | 0.634807 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321458 | 0.320976 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.236369 | 0.256901 | 0.92× | +8.7% | regression | no |
| vector_add | 67108864 | 512 | 0.250720 | 0.269025 | 0.93× | +7.3% | regression | no |
