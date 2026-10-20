# Benchmark comparison

NVIDIA H200 → NVIDIA GB10

Statistic: **mean**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 67.

CI gate: **FAIL / not eligible**.

- Gate: GPU metadata differs.
- Gate: 4 matched scenarios exceed the CV limit (0.1).
- Gate: 67 matched scenarios exceed the slowdown threshold.
- Different GPUs; ratios describe these runs, not an isolated hardware speedup or a CI regression.
- Run metadata differs: benchmark_binary_sha256.
- Run metadata differs: runtime_environment.

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008535 | 0.008166 | 1.05× | -4.3% | within_threshold | yes |
| gemm_naive | 256 | 16 | 0.013607 | 0.027951 | 0.49× | +105.4% | regression | no |
| gemm_naive | 512 | 16 | 0.060509 | 0.153260 | 0.39× | +153.3% | regression | no |
| gemm_naive | 1024 | 16 | 0.418365 | 1.157082 | 0.36× | +176.6% | regression | no |
| gemm_naive | 2048 | 16 | 3.177724 | 9.158740 | 0.35× | +188.2% | regression | no |
| gemm_tiled | 128 | 16 | 0.007400 | 0.008236 | 0.90× | +11.3% | regression | yes |
| gemm_tiled | 256 | 16 | 0.011638 | 0.019965 | 0.58× | +71.6% | regression | no |
| gemm_tiled | 512 | 16 | 0.042098 | 0.116913 | 0.36× | +177.7% | regression | no |
| gemm_tiled | 1024 | 16 | 0.279204 | 0.869273 | 0.32× | +211.3% | regression | no |
| gemm_tiled | 2048 | 16 | 2.106435 | 6.836049 | 0.31× | +224.5% | regression | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014849 | 0.014696 | 1.01× | -1.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009735 | 0.009648 | 1.01× | -0.9% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007283 | 0.009574 | 0.76× | +31.5% | regression | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006660 | 0.009825 | 0.68× | +47.5% | regression | yes |
| memcpy_bandwidth | 4194304 | 64 | 0.044270 | 0.117629 | 0.38× | +165.7% | regression | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024650 | 0.126033 | 0.20× | +411.3% | regression | no |
| memcpy_bandwidth | 4194304 | 256 | 0.014870 | 0.116908 | 0.13× | +686.2% | regression | no |
| memcpy_bandwidth | 4194304 | 512 | 0.012328 | 0.118239 | 0.10× | +859.1% | regression | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162387 | 0.541073 | 0.30× | +233.2% | regression | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084024 | 0.542607 | 0.15× | +545.8% | regression | no |
| memcpy_bandwidth | 16777216 | 256 | 0.054234 | 0.542368 | 0.10× | +900.1% | regression | no |
| memcpy_bandwidth | 16777216 | 512 | 0.058014 | 0.541174 | 0.11× | +832.8% | regression | no |
| memcpy_bandwidth | 67108864 | 64 | 0.634991 | 2.190472 | 0.29× | +245.0% | regression | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320925 | 2.189617 | 0.15× | +582.3% | regression | no |
| memcpy_bandwidth | 67108864 | 256 | 0.201066 | 2.188559 | 0.09× | +988.5% | regression | no |
| memcpy_bandwidth | 67108864 | 512 | 0.217326 | 2.172956 | 0.10× | +899.9% | regression | no |
| reduction | 1048576 | 64 | 0.015269 | 0.019089 | 0.80× | +25.0% | regression | no |
| reduction | 1048576 | 128 | 0.010479 | 0.019457 | 0.54× | +85.7% | regression | no |
| reduction | 1048576 | 256 | 0.009504 | 0.020230 | 0.47× | +112.9% | regression | no |
| reduction | 1048576 | 512 | 0.009951 | 0.022342 | 0.45× | +124.5% | regression | no |
| reduction | 4194304 | 64 | 0.044942 | 0.062908 | 0.71× | +40.0% | regression | no |
| reduction | 4194304 | 128 | 0.025322 | 0.063025 | 0.40× | +148.9% | regression | no |
| reduction | 4194304 | 256 | 0.021955 | 0.066717 | 0.33× | +203.9% | regression | no |
| reduction | 4194304 | 512 | 0.023862 | 0.075109 | 0.32× | +214.8% | regression | no |
| reduction | 16777216 | 64 | 0.163048 | 0.265197 | 0.61× | +62.6% | regression | no |
| reduction | 16777216 | 128 | 0.084775 | 0.269062 | 0.32× | +217.4% | regression | no |
| reduction | 16777216 | 256 | 0.080980 | 0.291770 | 0.28× | +260.3% | regression | no |
| reduction | 16777216 | 512 | 0.091471 | 0.338577 | 0.27× | +270.1% | regression | no |
| reduction | 67108864 | 64 | 0.635512 | 1.089135 | 0.58× | +71.4% | regression | no |
| reduction | 67108864 | 128 | 0.321372 | 1.104559 | 0.29× | +243.7% | regression | no |
| reduction | 67108864 | 256 | 0.305980 | 1.193149 | 0.26× | +289.9% | regression | no |
| reduction | 67108864 | 512 | 0.347863 | 1.389361 | 0.25× | +299.4% | regression | no |
| stencil_1d | 1048576 | 64 | 0.014855 | 0.014484 | 1.03× | -2.5% | within_threshold | no |
| stencil_1d | 1048576 | 128 | 0.009919 | 0.009823 | 1.01× | -1.0% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007604 | 0.009565 | 0.79× | +25.8% | regression | no |
| stencil_1d | 1048576 | 512 | 0.006993 | 0.009633 | 0.73× | +37.8% | regression | no |
| stencil_1d | 4194304 | 64 | 0.044351 | 0.120897 | 0.37× | +172.6% | regression | no |
| stencil_1d | 4194304 | 128 | 0.024710 | 0.133730 | 0.18× | +441.2% | regression | no |
| stencil_1d | 4194304 | 256 | 0.015071 | 0.117170 | 0.13× | +677.4% | regression | no |
| stencil_1d | 4194304 | 512 | 0.013756 | 0.119844 | 0.11× | +771.2% | regression | no |
| stencil_1d | 16777216 | 64 | 0.162594 | 0.568639 | 0.29× | +249.7% | regression | no |
| stencil_1d | 16777216 | 128 | 0.084264 | 0.558877 | 0.15× | +563.2% | regression | no |
| stencil_1d | 16777216 | 256 | 0.058954 | 0.555300 | 0.11× | +841.9% | regression | no |
| stencil_1d | 16777216 | 512 | 0.063097 | 0.562532 | 0.11× | +791.5% | regression | no |
| stencil_1d | 67108864 | 64 | 0.635005 | 2.291542 | 0.28× | +260.9% | regression | no |
| stencil_1d | 67108864 | 128 | 0.321391 | 2.243622 | 0.14× | +598.1% | regression | no |
| stencil_1d | 67108864 | 256 | 0.219999 | 2.230273 | 0.10× | +913.8% | regression | no |
| stencil_1d | 67108864 | 512 | 0.236125 | 2.264730 | 0.10× | +859.1% | regression | no |
| vector_add | 1048576 | 64 | 0.014818 | 0.014431 | 1.03× | -2.6% | within_threshold | yes |
| vector_add | 1048576 | 128 | 0.009742 | 0.009674 | 1.01× | -0.7% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007460 | 0.009747 | 0.77× | +30.7% | regression | no |
| vector_add | 1048576 | 512 | 0.006682 | 0.009687 | 0.69× | +45.0% | regression | no |
| vector_add | 4194304 | 64 | 0.044570 | 0.192698 | 0.23× | +332.3% | regression | no |
| vector_add | 4194304 | 128 | 0.025100 | 0.188867 | 0.13× | +652.5% | regression | no |
| vector_add | 4194304 | 256 | 0.019192 | 0.196966 | 0.10× | +926.3% | regression | no |
| vector_add | 4194304 | 512 | 0.020295 | 0.192236 | 0.11× | +847.2% | regression | no |
| vector_add | 16777216 | 64 | 0.162735 | 0.814123 | 0.20× | +400.3% | regression | no |
| vector_add | 16777216 | 128 | 0.084349 | 0.822870 | 0.10× | +875.5% | regression | no |
| vector_add | 16777216 | 256 | 0.062930 | 0.817535 | 0.08× | +1199.1% | regression | no |
| vector_add | 16777216 | 512 | 0.066664 | 0.814451 | 0.08× | +1121.7% | regression | no |
| vector_add | 67108864 | 64 | 0.634989 | 3.281283 | 0.19× | +416.7% | regression | no |
| vector_add | 67108864 | 128 | 0.321458 | 3.295358 | 0.10× | +925.1% | regression | no |
| vector_add | 67108864 | 256 | 0.236369 | 3.274914 | 0.07× | +1285.5% | regression | no |
| vector_add | 67108864 | 512 | 0.250720 | 3.278876 | 0.08× | +1207.8% | regression | no |
