# Benchmark comparison

NVIDIA H200 → NVIDIA GB10

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 66.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 2 matched scenarios exceed the CV limit (0.1).
- Gate: 66 matched scenarios exceed the slowdown threshold.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008694 | 0.008015 | 1.08× | -7.8% | improvement | yes |
| gemm_naive | 256 | 16 | 0.013634 | 0.027832 | 0.49× | +104.1% | regression | no |
| gemm_naive | 512 | 16 | 0.060617 | 0.153235 | 0.40× | +152.8% | regression | no |
| gemm_naive | 1024 | 16 | 0.418902 | 1.153489 | 0.36× | +175.4% | regression | no |
| gemm_naive | 2048 | 16 | 3.178079 | 9.171695 | 0.35× | +188.6% | regression | no |
| gemm_tiled | 128 | 16 | 0.007561 | 0.007268 | 1.04× | -3.9% | within_threshold | no |
| gemm_tiled | 256 | 16 | 0.011595 | 0.019961 | 0.58× | +72.2% | regression | no |
| gemm_tiled | 512 | 16 | 0.042239 | 0.115766 | 0.36× | +174.1% | regression | no |
| gemm_tiled | 1024 | 16 | 0.279226 | 0.869060 | 0.32× | +211.2% | regression | no |
| gemm_tiled | 2048 | 16 | 2.106378 | 6.851695 | 0.31× | +225.3% | regression | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014842 | 0.014259 | 1.04× | -3.9% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009973 | 0.009784 | 1.02× | -1.9% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007538 | 0.009629 | 0.78× | +27.7% | regression | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006582 | 0.009964 | 0.66× | +51.4% | regression | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044479 | 0.111846 | 0.40× | +151.5% | regression | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024771 | 0.118175 | 0.21× | +377.1% | regression | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015134 | 0.121402 | 0.12× | +702.2% | regression | no |
| memcpy_bandwidth | 4194304 | 512 | 0.012598 | 0.118774 | 0.11× | +842.8% | regression | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162600 | 0.546999 | 0.30× | +236.4% | regression | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084113 | 0.543862 | 0.15× | +546.6% | regression | no |
| memcpy_bandwidth | 16777216 | 256 | 0.054241 | 0.542875 | 0.10× | +900.9% | regression | no |
| memcpy_bandwidth | 16777216 | 512 | 0.058279 | 0.542221 | 0.11× | +830.4% | regression | no |
| memcpy_bandwidth | 67108864 | 64 | 0.634971 | 2.167763 | 0.29× | +241.4% | regression | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320955 | 2.185547 | 0.15× | +581.0% | regression | no |
| memcpy_bandwidth | 67108864 | 256 | 0.201094 | 2.190357 | 0.09× | +989.2% | regression | no |
| memcpy_bandwidth | 67108864 | 512 | 0.217239 | 2.192287 | 0.10× | +909.2% | regression | no |
| reduction | 1048576 | 64 | 0.015482 | 0.018826 | 0.82× | +21.6% | regression | no |
| reduction | 1048576 | 128 | 0.010612 | 0.019357 | 0.55× | +82.4% | regression | no |
| reduction | 1048576 | 256 | 0.009644 | 0.020185 | 0.48× | +109.3% | regression | no |
| reduction | 1048576 | 512 | 0.010042 | 0.022302 | 0.45× | +122.1% | regression | no |
| reduction | 4194304 | 64 | 0.044819 | 0.062758 | 0.71× | +40.0% | regression | no |
| reduction | 4194304 | 128 | 0.025306 | 0.062984 | 0.40× | +148.9% | regression | no |
| reduction | 4194304 | 256 | 0.022034 | 0.066854 | 0.33× | +203.4% | regression | no |
| reduction | 4194304 | 512 | 0.023987 | 0.075026 | 0.32× | +212.8% | regression | no |
| reduction | 16777216 | 64 | 0.163157 | 0.264444 | 0.62× | +62.1% | regression | no |
| reduction | 16777216 | 128 | 0.084747 | 0.268806 | 0.32× | +217.2% | regression | no |
| reduction | 16777216 | 256 | 0.080777 | 0.291740 | 0.28× | +261.2% | regression | no |
| reduction | 16777216 | 512 | 0.091473 | 0.339849 | 0.27× | +271.5% | regression | no |
| reduction | 67108864 | 64 | 0.635468 | 1.089985 | 0.58× | +71.5% | regression | no |
| reduction | 67108864 | 128 | 0.321361 | 1.115815 | 0.29× | +247.2% | regression | no |
| reduction | 67108864 | 256 | 0.305913 | 1.191097 | 0.26× | +289.4% | regression | no |
| reduction | 67108864 | 512 | 0.347927 | 1.374375 | 0.25× | +295.0% | regression | no |
| stencil_1d | 1048576 | 64 | 0.014966 | 0.013987 | 1.07× | -6.5% | improvement | no |
| stencil_1d | 1048576 | 128 | 0.009804 | 0.009788 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007581 | 0.009587 | 0.79× | +26.5% | regression | no |
| stencil_1d | 1048576 | 512 | 0.007058 | 0.009766 | 0.72× | +38.4% | regression | no |
| stencil_1d | 4194304 | 64 | 0.044439 | 0.118692 | 0.37× | +167.1% | regression | no |
| stencil_1d | 4194304 | 128 | 0.024613 | 0.121324 | 0.20× | +392.9% | regression | no |
| stencil_1d | 4194304 | 256 | 0.015019 | 0.126961 | 0.12× | +745.3% | regression | no |
| stencil_1d | 4194304 | 512 | 0.013874 | 0.126544 | 0.11× | +812.1% | regression | no |
| stencil_1d | 16777216 | 64 | 0.162577 | 0.566898 | 0.29× | +248.7% | regression | no |
| stencil_1d | 16777216 | 128 | 0.084388 | 0.558092 | 0.15× | +561.3% | regression | no |
| stencil_1d | 16777216 | 256 | 0.059060 | 0.552471 | 0.11× | +835.4% | regression | no |
| stencil_1d | 16777216 | 512 | 0.063128 | 0.571448 | 0.11× | +805.2% | regression | no |
| stencil_1d | 67108864 | 64 | 0.635112 | 2.281128 | 0.28× | +259.2% | regression | no |
| stencil_1d | 67108864 | 128 | 0.321555 | 2.248722 | 0.14× | +599.3% | regression | no |
| stencil_1d | 67108864 | 256 | 0.220111 | 2.232627 | 0.10× | +914.3% | regression | no |
| stencil_1d | 67108864 | 512 | 0.236063 | 2.271428 | 0.10× | +862.2% | regression | no |
| vector_add | 1048576 | 64 | 0.014817 | 0.013959 | 1.06× | -5.8% | improvement | no |
| vector_add | 1048576 | 128 | 0.009949 | 0.009727 | 1.02× | -2.2% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007450 | 0.009966 | 0.75× | +33.8% | regression | no |
| vector_add | 1048576 | 512 | 0.006885 | 0.010262 | 0.67× | +49.0% | regression | yes |
| vector_add | 4194304 | 64 | 0.044614 | 0.193820 | 0.23× | +334.4% | regression | no |
| vector_add | 4194304 | 128 | 0.025012 | 0.198026 | 0.13× | +691.7% | regression | no |
| vector_add | 4194304 | 256 | 0.019358 | 0.195869 | 0.10× | +911.8% | regression | no |
| vector_add | 4194304 | 512 | 0.020317 | 0.195261 | 0.10× | +861.1% | regression | no |
| vector_add | 16777216 | 64 | 0.162736 | 0.831861 | 0.20× | +411.2% | regression | no |
| vector_add | 16777216 | 128 | 0.084443 | 0.822653 | 0.10× | +874.2% | regression | no |
| vector_add | 16777216 | 256 | 0.063085 | 0.823995 | 0.08× | +1206.2% | regression | no |
| vector_add | 16777216 | 512 | 0.066663 | 0.825758 | 0.08× | +1138.7% | regression | no |
| vector_add | 67108864 | 64 | 0.635033 | 3.248033 | 0.20× | +411.5% | regression | no |
| vector_add | 67108864 | 128 | 0.321564 | 3.260865 | 0.10× | +914.1% | regression | no |
| vector_add | 67108864 | 256 | 0.236526 | 3.279658 | 0.07× | +1286.6% | regression | no |
| vector_add | 67108864 | 512 | 0.250900 | 3.247712 | 0.08× | +1194.4% | regression | no |
