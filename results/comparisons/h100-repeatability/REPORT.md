# Benchmark comparison

NVIDIA H100 80GB HBM3 → NVIDIA H100 80GB HBM3

Statistic: **median**. Threshold: **5%**. Speedup = baseline / candidate runtime; positive change means slower.

These are descriptive timing comparisons, not statistical significance tests. CV flags within-run variability; it does not measure uncertainty across independent runs.

Matched: 74; missing: 0; added: 0; regressions: 0.

CI gate: **FAIL / not eligible**.

- Gate: 1 matched scenarios exceed the CV limit (0.1).

| Kernel | Size | Block | Baseline ms | Candidate ms | Speedup | Change | Status | Noisy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| gemm_naive | 128 | 16 | 0.008768 | 0.008736 | 1.00× | -0.4% | within_threshold | no |
| gemm_naive | 256 | 16 | 0.013536 | 0.013584 | 1.00× | +0.4% | within_threshold | no |
| gemm_naive | 512 | 16 | 0.060208 | 0.060304 | 1.00× | +0.2% | within_threshold | no |
| gemm_naive | 1024 | 16 | 0.417888 | 0.417440 | 1.00× | -0.1% | within_threshold | no |
| gemm_naive | 2048 | 16 | 3.152816 | 3.176848 | 0.99× | +0.8% | within_threshold | no |
| gemm_tiled | 128 | 16 | 0.007632 | 0.007568 | 1.01× | -0.8% | within_threshold | no |
| gemm_tiled | 256 | 16 | 0.011552 | 0.011600 | 1.00× | +0.4% | within_threshold | no |
| gemm_tiled | 512 | 16 | 0.041808 | 0.042032 | 0.99× | +0.5% | within_threshold | no |
| gemm_tiled | 1024 | 16 | 0.276832 | 0.279024 | 0.99× | +0.8% | within_threshold | no |
| gemm_tiled | 2048 | 16 | 2.090160 | 2.105776 | 0.99× | +0.7% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 64 | 0.014752 | 0.014704 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 128 | 0.009824 | 0.009824 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 256 | 0.007328 | 0.007408 | 0.99× | +1.1% | within_threshold | no |
| memcpy_bandwidth | 1048576 | 512 | 0.006656 | 0.006592 | 1.01× | -1.0% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 64 | 0.044416 | 0.044352 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 128 | 0.024672 | 0.024704 | 1.00× | +0.1% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 256 | 0.015024 | 0.014976 | 1.00× | -0.3% | within_threshold | no |
| memcpy_bandwidth | 4194304 | 512 | 0.013712 | 0.013792 | 0.99× | +0.6% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 64 | 0.162752 | 0.162592 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 128 | 0.084080 | 0.084112 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 256 | 0.060576 | 0.060544 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 16777216 | 512 | 0.064928 | 0.064848 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 64 | 0.635168 | 0.635216 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 128 | 0.320832 | 0.320896 | 1.00× | +0.0% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 256 | 0.225600 | 0.225312 | 1.00× | -0.1% | within_threshold | no |
| memcpy_bandwidth | 67108864 | 512 | 0.243312 | 0.242576 | 1.00× | -0.3% | within_threshold | no |
| reduction | 1048576 | 64 | 0.015312 | 0.015424 | 0.99× | +0.7% | within_threshold | no |
| reduction | 1048576 | 128 | 0.010496 | 0.010528 | 1.00× | +0.3% | within_threshold | no |
| reduction | 1048576 | 256 | 0.009472 | 0.009472 | 1.00× | +0.0% | within_threshold | no |
| reduction | 1048576 | 512 | 0.010080 | 0.010064 | 1.00× | -0.2% | within_threshold | no |
| reduction | 4194304 | 64 | 0.044896 | 0.044864 | 1.00× | -0.1% | within_threshold | no |
| reduction | 4194304 | 128 | 0.025360 | 0.025312 | 1.00× | -0.2% | within_threshold | no |
| reduction | 4194304 | 256 | 0.021920 | 0.021888 | 1.00× | -0.1% | within_threshold | no |
| reduction | 4194304 | 512 | 0.024000 | 0.023984 | 1.00× | -0.1% | within_threshold | no |
| reduction | 16777216 | 64 | 0.163168 | 0.163072 | 1.00× | -0.1% | within_threshold | no |
| reduction | 16777216 | 128 | 0.084736 | 0.084544 | 1.00× | -0.2% | within_threshold | no |
| reduction | 16777216 | 256 | 0.081312 | 0.081216 | 1.00× | -0.1% | within_threshold | no |
| reduction | 16777216 | 512 | 0.091968 | 0.091888 | 1.00× | -0.1% | within_threshold | no |
| reduction | 67108864 | 64 | 0.636288 | 0.636352 | 1.00× | +0.0% | within_threshold | no |
| reduction | 67108864 | 128 | 0.321472 | 0.321376 | 1.00× | -0.0% | within_threshold | no |
| reduction | 67108864 | 256 | 0.307936 | 0.307904 | 1.00× | -0.0% | within_threshold | no |
| reduction | 67108864 | 512 | 0.350112 | 0.349984 | 1.00× | -0.0% | within_threshold | no |
| stencil_1d | 1048576 | 64 | 0.014816 | 0.014832 | 1.00× | +0.1% | within_threshold | no |
| stencil_1d | 1048576 | 128 | 0.009888 | 0.009888 | 1.00× | +0.0% | within_threshold | no |
| stencil_1d | 1048576 | 256 | 0.007456 | 0.007488 | 1.00× | +0.4% | within_threshold | no |
| stencil_1d | 1048576 | 512 | 0.006960 | 0.006944 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 4194304 | 64 | 0.044432 | 0.044368 | 1.00× | -0.1% | within_threshold | no |
| stencil_1d | 4194304 | 128 | 0.024832 | 0.024688 | 1.01× | -0.6% | within_threshold | no |
| stencil_1d | 4194304 | 256 | 0.015520 | 0.015376 | 1.01× | -0.9% | within_threshold | no |
| stencil_1d | 4194304 | 512 | 0.015680 | 0.015424 | 1.02× | -1.6% | within_threshold | no |
| stencil_1d | 16777216 | 64 | 0.162720 | 0.162864 | 1.00× | +0.1% | within_threshold | no |
| stencil_1d | 16777216 | 128 | 0.084352 | 0.084432 | 1.00× | +0.1% | within_threshold | no |
| stencil_1d | 16777216 | 256 | 0.065184 | 0.065040 | 1.00× | -0.2% | within_threshold | no |
| stencil_1d | 16777216 | 512 | 0.069312 | 0.069072 | 1.00× | -0.3% | within_threshold | no |
| stencil_1d | 67108864 | 64 | 0.635216 | 0.635424 | 1.00× | +0.0% | within_threshold | no |
| stencil_1d | 67108864 | 128 | 0.321312 | 0.321312 | 1.00× | +0.0% | within_threshold | no |
| stencil_1d | 67108864 | 256 | 0.243600 | 0.243712 | 1.00× | +0.0% | within_threshold | no |
| stencil_1d | 67108864 | 512 | 0.259968 | 0.259952 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 1048576 | 64 | 0.014720 | 0.014784 | 1.00× | +0.4% | within_threshold | no |
| vector_add | 1048576 | 128 | 0.009888 | 0.009824 | 1.01× | -0.6% | within_threshold | no |
| vector_add | 1048576 | 256 | 0.007392 | 0.007360 | 1.00× | -0.4% | within_threshold | yes |
| vector_add | 1048576 | 512 | 0.006752 | 0.006832 | 0.99× | +1.2% | within_threshold | no |
| vector_add | 4194304 | 64 | 0.044544 | 0.044576 | 1.00× | +0.1% | within_threshold | no |
| vector_add | 4194304 | 128 | 0.025072 | 0.025024 | 1.00× | -0.2% | within_threshold | no |
| vector_add | 4194304 | 256 | 0.023072 | 0.023040 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 4194304 | 512 | 0.023584 | 0.023632 | 1.00× | +0.2% | within_threshold | no |
| vector_add | 16777216 | 64 | 0.162624 | 0.162592 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 16777216 | 128 | 0.084544 | 0.084576 | 1.00× | +0.0% | within_threshold | no |
| vector_add | 16777216 | 256 | 0.077184 | 0.077072 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 16777216 | 512 | 0.079264 | 0.079296 | 1.00× | +0.0% | within_threshold | no |
| vector_add | 67108864 | 64 | 0.635184 | 0.635072 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 128 | 0.321824 | 0.321696 | 1.00× | -0.0% | within_threshold | no |
| vector_add | 67108864 | 256 | 0.290720 | 0.290464 | 1.00× | -0.1% | within_threshold | no |
| vector_add | 67108864 | 512 | 0.298576 | 0.298496 | 1.00× | -0.0% | within_threshold | no |
