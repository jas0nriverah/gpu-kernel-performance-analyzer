# Benchmark comparison

NVIDIA H100 80GB HBM3 → NVIDIA H200

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 0.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: git_commit.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.
- Run metadata differs: source_sha256.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008832 | 0.008694 | 1.02× | -1.6% | within_threshold | no |
| gemm_naive | 256 | 16 | 0.013674 | 0.013634 | 1.00× | -0.3% | within_threshold | no |
| gemm_naive | 512 | 16 | 0.060282 | 0.060617 | 0.99× | +0.6% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.418256 | 0.418902 | 1.00× | +0.2% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.159608 | 3.178079 | 0.99× | +0.6% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007690 | 0.007561 | 1.02× | -1.7% | within_threshold | no |
| gemm_tiled | 256 | 16 | 0.011566 | 0.011595 | 1.00× | +0.2% | within_threshold | no |
| gemm_tiled | 512 | 16 | 0.041894 | 0.042239 | 0.99× | +0.8% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.276892 | 0.279226 | 0.99× | +0.8% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.090242 | 2.106378 | 0.99× | +0.8% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014817 | 0.014842 | 1.00× | +0.2% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009880 | 0.009973 | 0.99× | +0.9% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007391 | 0.007538 | 0.98× | +2.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006709 | 0.006582 | 1.02× | -1.9% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044471 | 0.044479 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024703 | 0.024771 | 1.00× | +0.3% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015071 | 0.015134 | 1.00× | +0.4% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013736 | 0.012598 | 1.09× | -8.3% | improvement | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162840 | 0.162600 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084141 | 0.084113 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060476 | 0.054241 | 1.11× | -10.3% | improvement | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064798 | 0.058279 | 1.11× | -10.1% | improvement | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635199 | 0.634971 | 1.00× | -0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320904 | 0.320955 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225754 | 0.201094 | 1.12× | -10.9% | improvement | no |
| memcpy_bandwidth | 67108864 | 512 | 0.243235 | 0.217239 | 1.12× | -10.7% | improvement | no |
| reduction | 1048576 | 64 | 0.015467 | 0.015482 | 1.00× | +0.1% | within_threshold | no |
| reduction | 1048576 | 128 | 0.010579 | 0.010612 | 1.00× | +0.3% | within_threshold | no |
| reduction | 1048576 | 256 | 0.009557 | 0.009644 | 0.99× | +0.9% | within_threshold | no |
| reduction | 1048576 | 512 | 0.010067 | 0.010042 | 1.00× | -0.2% | within_threshold | no |
| reduction | 4194304 | 64 | 0.044980 | 0.044819 | 1.00× | -0.4% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025431 | 0.025306 | 1.00× | -0.5% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021996 | 0.022034 | 1.00× | +0.2% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024077 | 0.023987 | 1.00× | -0.4% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163196 | 0.163157 | 1.00× | -0.0% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084748 | 0.084747 | 1.00× | -0.0% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081283 | 0.080777 | 1.01× | -0.6% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091993 | 0.091473 | 1.01× | -0.6% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636312 | 0.635468 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321516 | 0.321361 | 1.00× | -0.0% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307970 | 0.305913 | 1.01× | -0.7% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350140 | 0.347927 | 1.01× | -0.6% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014930 | 0.014966 | 1.00× | +0.2% | within_threshold | no |
| stencil_1d | 1048576 | 128 | 0.009933 | 0.009804 | 1.01× | -1.3% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007495 | 0.007581 | 0.99× | +1.2% | within_threshold | no |
| stencil_1d | 1048576 | 512 | 0.007018 | 0.007058 | 0.99× | +0.6% | within_threshold | no |
| stencil_1d | 4194304 | 64 | 0.044474 | 0.044439 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.025047 | 0.024613 | 1.02× | -1.7% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015590 | 0.015019 | 1.04× | -3.7% | within_threshold | no |
| stencil_1d | 4194304 | 512 | 0.015658 | 0.013874 | 1.13× | -11.4% | improvement | no |
| stencil_1d | 16777216 | 64 | 0.162772 | 0.162577 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084416 | 0.084388 | 1.00× | -0.0% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.065088 | 0.059060 | 1.10× | -9.3% | improvement | no |
| stencil_1d | 16777216 | 512 | 0.069218 | 0.063128 | 1.10× | -8.8% | improvement | no |
| stencil_1d | 67108864 | 64 | 0.635241 | 0.635112 | 1.00× | -0.0% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321335 | 0.321555 | 1.00× | +0.1% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243750 | 0.220111 | 1.11× | -9.7% | improvement | no |
| stencil_1d | 67108864 | 512 | 0.259904 | 0.236063 | 1.10× | -9.2% | improvement | no |
| vector_add | 1048576 | 64 | 0.014799 | 0.014817 | 1.00× | +0.1% | within_threshold | no |
| vector_add | 1048576 | 128 | 0.009915 | 0.009949 | 1.00× | +0.3% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007434 | 0.007450 | 1.00× | +0.2% | within_threshold | no |
| vector_add | 1048576 | 512 | 0.006800 | 0.006885 | 0.99× | +1.3% | within_threshold | no |
| vector_add | 4194304 | 64 | 0.044617 | 0.044614 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025122 | 0.025012 | 1.00× | -0.4% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023203 | 0.019358 | 1.20× | -16.6% | improvement | no |
| vector_add | 4194304 | 512 | 0.023811 | 0.020317 | 1.17× | -14.7% | improvement | no |
| vector_add | 16777216 | 64 | 0.162646 | 0.162736 | 1.00× | +0.1% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084497 | 0.084443 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.076991 | 0.063085 | 1.22× | -18.1% | improvement | no |
| vector_add | 16777216 | 512 | 0.079134 | 0.066663 | 1.19× | -15.8% | improvement | no |
| vector_add | 67108864 | 64 | 0.635240 | 0.635033 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321926 | 0.321564 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290634 | 0.236526 | 1.23× | -18.6% | improvement | no |
| vector_add | 67108864 | 512 | 0.298547 | 0.250900 | 1.19× | -16.0% | improvement | no |
