# GPU Kernel Performance Report

## Run Manifest Snapshot
```json
{
  "benchmark_binary": "/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/build/gpu_benchmark",
  "benchmark_binary_sha256": "e592bb4171a1c4d46dab2368920936de3a0ff922be5798b8b1d0c83d0f1ad0f9",
  "created_at_utc": "2026-09-22T23:20:37.806330+00:00",
  "device": {
    "compute_capability_major": 9,
    "compute_capability_minor": 0,
    "metadata_available": true,
    "multiprocessor_count": 132,
    "name": "NVIDIA H100 80GB HBM3",
    "total_global_mem_bytes": 85019590656
  },
  "git_commit": "31388b9dca9099fd2d229bfa4520ab2e7b45f590",
  "metrics_policy_version": "mvp_v1",
  "nsight_compute": {
    "available": true,
    "enabled": true,
    "import_timestamp": "2026-09-22T23:27:56.215768+00:00",
    "imported_metric_rows": 5,
    "imported_metric_rows_total": 10,
    "imports": [
      {
        "import_timestamp": "2026-09-22T23:27:55.121350+00:00",
        "imported_metric_rows": 5,
        "metric_set": "default_profiler_set",
        "source_csv": "/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/results/h100-2026-09-22/profiler/vector_add.csv",
        "source_file": "results/h100-2026-09-22/profiler/vector_add_raw.csv",
        "source_tool": "ncu"
      },
      {
        "import_timestamp": "2026-09-22T23:27:56.215768+00:00",
        "imported_metric_rows": 5,
        "metric_set": "default_profiler_set",
        "source_csv": "/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/results/h100-2026-09-22/profiler/gemm_tiled.csv",
        "source_file": "results/h100-2026-09-22/profiler/gemm_tiled_raw.csv",
        "source_tool": "ncu"
      }
    ],
    "metric_set": "default_profiler_set",
    "source_csv": "/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/results/h100-2026-09-22/profiler/gemm_tiled.csv",
    "source_file": "results/h100-2026-09-22/profiler/gemm_tiled_raw.csv",
    "source_tool": "ncu",
    "status": "parsed_metrics"
  },
  "run_id": "run_20260922T232026Z",
  "runtime_environment": {
    "machine": "x86_64",
    "platform": "Linux-5.14.0-570.128.1.el9_6.x86_64-x86_64-with-glibc2.34",
    "python_version": "3.11.9"
  },
  "scenario_count": 24,
  "scenario_file": "/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/configs/benchmark_scenarios.yaml",
  "scenario_file_sha256": "939c299649d41c1bf8d425897373171110afbfc759d265f83a3ba6326cd0b012",
  "source_sha256": {
    "benchmarks/CMakeLists.txt": "ceae6d75ccf2d56082fa63def2ea3513a9f275ceb1e2560a580de8514fb7e8e9",
    "benchmarks/include/benchmark_runner.h": "0889eca0572d793d629911dde5eda3451610304bc38b45759ea4ff30cdc50a68",
    "benchmarks/include/benchmark_types.h": "b9336f02dd9b3762329477352f07db8f07bc2be22328871b279d1cda5730c55a",
    "benchmarks/include/cuda_utils.h": "ac2e6bb575e1b2d13e248de3b9823b24bad3e042581a4213386015a3049ba62d",
    "benchmarks/include/kernel_ops.h": "c4c358faa52276606eac0906f8fdf7739df6d5a05d415286381af09e7704e883",
    "benchmarks/src/benchmark_runner.cu": "625f2342fd30f2e5c357569456ad15dccf88c53e095f10dc86eda2a6206009c9",
    "benchmarks/src/cuda_utils.cu": "43d851be8f86c65aee1fbf06261c470f3de71586221b39c4f2e0c5f7223f5239",
    "benchmarks/src/kernels/gemm_naive.cu": "67729ad789877492c01fd8f06e5fe937a0b7105a83ba0e45140688bf4db5c8f3",
    "benchmarks/src/kernels/gemm_tiled.cu": "bbb39ec587516a462adad4fd666b0ed24c3ab2f9021f703898966e5e392b67dc",
    "benchmarks/src/kernels/memcpy_bandwidth.cu": "789d0ebc8e0e609553f8af5af99f90be306b496211016033ba61d642722c36db",
    "benchmarks/src/kernels/reduction.cu": "563ce1159a0e4db1d67d84b24ffb40f8e8f46e0ded8720dcef1f40a18e2c346c",
    "benchmarks/src/kernels/stencil_1d.cu": "5f3bf1e0ebca5c59b3dd0ecbef635d2946fd5572f8f612c13a9dcb9d2662f3fb",
    "benchmarks/src/kernels/vector_add.cu": "203342f0a5677e137b460a0941b94470ee04327ed429f3ae3a1636962ea1099a",
    "benchmarks/src/main.cu": "2e3ac45cea731b4c59ddea6ed4cc3b81ef02db0c91ba32f51904184aa0cf75cc",
    "src/gpu_kernel_analyzer/__init__.py": "f8dbbda778b3a0da3f3ff01b590cbf656f9315015ed01dc6e8fc23e9bc2ca9f7",
    "src/gpu_kernel_analyzer/__main__.py": "3129adff95e0f38b87ab34cddfc27082c7892bcaec267ff5d40f160fdf55b44a",
    "src/gpu_kernel_analyzer/advisor.py": "18f7bccfd2b7d6f93213ac6020955e371b7acc1ebb07eae37d34e3d5d4cce228",
    "src/gpu_kernel_analyzer/analysis.py": "2bc5fb9b07335fee524647047deac20c6c6ca96ec293f69e07d9ef46a950ddd9",
    "src/gpu_kernel_analyzer/artifacts.py": "58a00ff917586295feb85bf75340836f65a8c53174cab46ff1607b9e8784c8bd",
    "src/gpu_kernel_analyzer/cli.py": "5267c7dce41cacd703ed474fe689934b5a8fa7a4ac6c1897bc9f69c6c97c5b13",
    "src/gpu_kernel_analyzer/io.py": "56dcb6e4e0cabe4434a43deab926b21460b94eb11ddc348f6abcf75f3b0b1698",
    "src/gpu_kernel_analyzer/metrics.py": "03a0648f899d7765327f614ce75cda44bd7b0ab1af64ccd61fcd31a081482b57",
    "src/gpu_kernel_analyzer/ncu.py": "09af1af41703b25f3a55d2ba4c7289c8ed2a8dd12fe0dd789ca5918716c529d5",
    "src/gpu_kernel_analyzer/perf_model.py": "57d4e68211745118af2dafb897d004119b806aabba2185e7994544d82b0000b1",
    "src/gpu_kernel_analyzer/plotting.py": "0c5674354eb99df518652af3d30ffa5b67e387b84f41759fccbb5db9df4eab91",
    "src/gpu_kernel_analyzer/report.py": "4c17784ded3436f281fb5471b21f74252e14e19521eec091c11a174322317988",
    "src/gpu_kernel_analyzer/runner.py": "2d80af3a84977204bcc890da2a6e96d35dc24363b0f98a3e1a5315803247820b",
    "src/gpu_kernel_analyzer/scenarios.py": "97dc8a160f30408913f158d64074b5a6653c70c5612300425e3e6534189fc728",
    "src/gpu_kernel_analyzer/schemas.py": "0d36a13d1835f63e49938ecd369b5d43b9bf6b5b1fb1448bddebf70024c67d6b",
    "src/gpu_kernel_analyzer/system_info.py": "853fd91be7da5126d38101b108a61c696160f3a9b9cebed75ac388340c5405eb"
  }
}
```

## Benchmark Summary
| kernel | problem_size | block_size | runtime_ms_mean | runtime_ms_min | runtime_ms_max | runtime_ms_cv | effective_bandwidth_GBps | effective_GFLOPs | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector_add | 1048576 | 64 | 0.014797866666666666 | 0.014336 | 0.015552 | 0.02077688669279788 | 850.3193253081525 | 70.85994377567938 | 0.08333333333333333 |
| vector_add | 1048576 | 128 | 0.009963733333333334 | 0.009664 | 0.01056 | 0.024367402263243895 | 1262.8712129322341 | 105.23926774435284 | 0.08333333333333333 |
| vector_add | 1048576 | 256 | 0.0074624 | 0.007136 | 0.008384 | 0.038184735923022094 | 1686.1749571183534 | 140.51457975986278 | 0.08333333333333333 |
| vector_add | 1048576 | 512 | 0.006871466666666667 | 0.006464 | 0.00752 | 0.03914661186243636 | 1831.182862465073 | 152.59857187208942 | 0.08333333333333333 |
| vector_add | 4194304 | 64 | 0.0446624 | 0.04416 | 0.045792 | 0.007425578263575945 | 1126.935587877051 | 93.91129898975424 | 0.08333333333333333 |
| vector_add | 4194304 | 128 | 0.025179733333333332 | 0.024736 | 0.025728 | 0.009989197839014755 | 1998.8951961365756 | 166.57459967804797 | 0.08333333333333333 |
| vector_add | 4194304 | 256 | 0.0232608 | 0.022976 | 0.023968 | 0.013638920523542814 | 2163.7969459347914 | 180.31641216123262 | 0.08333333333333333 |
| vector_add | 4194304 | 512 | 0.023768533333333335 | 0.02336 | 0.024672 | 0.014148868702975336 | 2117.574832832204 | 176.46456940268365 | 0.08333333333333333 |
| reduction | 1048576 | 64 | 0.015479466666666667 | 0.01504 | 0.016352 | 0.01848186837292195 | 275.19294377067257 | 67.73973694184123 | 0.24615361140324518 |
| reduction | 1048576 | 128 | 0.010541866666666667 | 0.010176 | 0.011264 | 0.020145753025052157 | 400.9794596782354 | 99.46767808357785 | 0.2480617789335029 |
| reduction | 1048576 | 256 | 0.0096256 | 0.009152 | 0.010688 | 0.03468085786233235 | 437.4468085106383 | 108.93606632313829 | 0.24902699986320526 |
| reduction | 1048576 | 512 | 0.010286933333333333 | 0.009728 | 0.012128 | 0.04542630803424145 | 408.52758191621734 | 101.93271075279966 | 0.2495124326114766 |
| reduction | 4194304 | 64 | 0.0449568 | 0.044512 | 0.046112 | 0.0074366907163439565 | 379.0163000925333 | 93.29629777920137 | 0.2461537874661959 |
| reduction | 4194304 | 128 | 0.025425066666666666 | 0.024928 | 0.028544 | 0.024901494362486492 | 665.0243329417689 | 164.9672370573922 | 0.2480619563612827 |
| reduction | 4194304 | 256 | 0.022027733333333334 | 0.021696 | 0.022752 | 0.010925874217775203 | 764.6157571061934 | 190.41010423224057 | 0.2490271779813655 |
| reduction | 4194304 | 512 | 0.0240576 | 0.02368 | 0.024864 | 0.011162622691170188 | 698.7390263367918 | 174.3441989225858 | 0.24951261107684575 |
| gemm_naive | 256 | 16 | 0.0138496 | 0.01344 | 0.014752 | 0.026588906182776612 | 56.78373382624769 | 2422.772643253235 | 42.666666666666664 |
| gemm_naive | 512 | 16 | 0.0605568 | 0.059776 | 0.061696 | 0.009084890252775058 | 51.94673430564362 | 4432.787994081589 | 85.33333333333333 |
| gemm_tiled | 256 | 16 | 0.011837866666666667 | 0.011264 | 0.013312 | 0.03938955530490521 | 66.433591638133 | 2834.4999098936746 | 42.666666666666664 |
| gemm_tiled | 512 | 16 | 0.042254933333333335 | 0.041664 | 0.043296 | 0.010114591381342875 | 74.44640783561367 | 6352.7601353057 | 85.33333333333333 |
| memcpy_bandwidth | 1048576 | 256 | 0.007514666666666667 | 0.00704 | 0.008384 | 0.043427869724921624 | 1116.2980837473385 | 0.0 | 0.0 |
| memcpy_bandwidth | 4194304 | 256 | 0.0151232 | 0.01472 | 0.016256 | 0.021043552874830302 | 2218.738891239949 | 0.0 | 0.0 |
| stencil_1d | 1048576 | 256 | 0.007611733333333333 | 0.007168 | 0.00832 | 0.03361009170462065 | 1102.0627802690585 | 688.7892376681615 | 0.625 |
| stencil_1d | 4194304 | 256 | 0.015460266666666667 | 0.015072 | 0.015968 | 0.01207770645074316 | 2170.3656685525048 | 1356.4785428453156 | 0.625 |

## Key Result
gemm_tiled at 512x512 ran about 1.43x faster than gemm_naive (about 6353 vs 4433 effective GFLOPs).

## Speedup vs Baseline
| optimized_kernel | baseline_kernel | problem_size | speedup_runtime | baseline_GFLOPs | optimized_GFLOPs |
| --- | --- | --- | --- | --- | --- |
| gemm_tiled | gemm_naive | 256 | 1.1699405298251937 | 2422.772643253235 | 2834.4999098936746 |
| gemm_tiled | gemm_naive | 512 | 1.4331297016206392 | 4432.787994081589 | 6352.7601353057 |

## Metric Integrity Notes
- `runtime_ms` is measured with CUDA events.
- `effective_bandwidth_GBps`, `effective_GFLOPs`, and `arithmetic_intensity` are derived estimates.
- Nsight Compute timing overhead is not used for benchmark `runtime_ms` claims.
- Profiler metrics are scenario-specific and only measured for scenarios with imported Nsight CSV rows.

## Metrics Provenance (Default Metrics)
| kernel | problem_size | block_size | metric_name | status | source | metric_value |
| --- | --- | --- | --- | --- | --- | --- |
| vector_add | 1048576 | 64 | runtime_ms | measured | cuda_events | 0.014797866666666666 |
| vector_add | 1048576 | 64 | effective_bandwidth_GBps | derived_estimate | bytes_moved/runtime_ms | 850.3193253081525 |
| vector_add | 1048576 | 64 | effective_GFLOPs | derived_estimate | flops/runtime_ms | 70.85994377567938 |
| vector_add | 1048576 | 64 | arithmetic_intensity | derived_estimate | flops/bytes_moved | 0.08333333333333333 |
| vector_add | 1048576 | 64 | device_metadata | measured | cuda_runtime_api | {"compute_capability_major": 9, "compute_capability_minor": 0, "metadata_available": true, "multiprocessor_count": 132, "name": "NVIDIA H100 80GB HBM3", "total_global_mem_bytes": 85019590656} |
| vector_add | 1048576 | 128 | runtime_ms | measured | cuda_events | 0.009963733333333334 |
| vector_add | 1048576 | 128 | effective_bandwidth_GBps | derived_estimate | bytes_moved/runtime_ms | 1262.8712129322341 |
| vector_add | 1048576 | 128 | effective_GFLOPs | derived_estimate | flops/runtime_ms | 105.23926774435284 |
| vector_add | 1048576 | 128 | arithmetic_intensity | derived_estimate | flops/bytes_moved | 0.08333333333333333 |
| vector_add | 1048576 | 128 | device_metadata | measured | cuda_runtime_api | {"compute_capability_major": 9, "compute_capability_minor": 0, "metadata_available": true, "multiprocessor_count": 132, "name": "NVIDIA H100 80GB HBM3", "total_global_mem_bytes": 85019590656} |

## Profiler Metrics Status
| kernel | problem_size | block_size | metric_name | status | metric_value | source |
| --- | --- | --- | --- | --- | --- | --- |
| vector_add | 1048576 | 64 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 64 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 64 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 64 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 64 | sm_utilization | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 128 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 128 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 128 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 128 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 128 | sm_utilization | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 512 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 512 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 512 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 512 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 1048576 | 512 | sm_utilization | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 64 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 64 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 64 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 64 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 64 | sm_utilization | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 128 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 128 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 128 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 128 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 128 | sm_utilization | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 256 | l2_cache_hit_rate | measured | 34.44 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/vector_add_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:55.121350+00:00 |
| vector_add | 4194304 | 256 | l2_throughput_pct | measured | 74.11 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/vector_add_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:55.121350+00:00 |
| vector_add | 4194304 | 256 | memory_throughput_pct | measured | 66.87 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/vector_add_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:55.121350+00:00 |
| vector_add | 4194304 | 256 | occupancy | measured | 74.98 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/vector_add_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:55.121350+00:00 |
| vector_add | 4194304 | 256 | sm_utilization | measured | 18.90 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/vector_add_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:55.121350+00:00 |
| vector_add | 4194304 | 512 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 512 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 512 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 512 | occupancy | unavailable |  | nsight_compute_not_run |
| vector_add | 4194304 | 512 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 64 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 64 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 64 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 64 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 64 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 128 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 128 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 128 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 128 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 128 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 512 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 512 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 512 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 512 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 1048576 | 512 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 64 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 64 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 64 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 64 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 64 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 128 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 128 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 128 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 128 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 128 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 512 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 512 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 512 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 512 | occupancy | unavailable |  | nsight_compute_not_run |
| reduction | 4194304 | 512 | sm_utilization | unavailable |  | nsight_compute_not_run |
| gemm_naive | 256 | 16 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| gemm_naive | 256 | 16 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| gemm_naive | 256 | 16 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| gemm_naive | 256 | 16 | occupancy | unavailable |  | nsight_compute_not_run |
| gemm_naive | 256 | 16 | sm_utilization | unavailable |  | nsight_compute_not_run |
| gemm_naive | 512 | 16 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| gemm_naive | 512 | 16 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| gemm_naive | 512 | 16 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| gemm_naive | 512 | 16 | occupancy | unavailable |  | nsight_compute_not_run |
| gemm_naive | 512 | 16 | sm_utilization | unavailable |  | nsight_compute_not_run |
| gemm_tiled | 256 | 16 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| gemm_tiled | 256 | 16 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| gemm_tiled | 256 | 16 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| gemm_tiled | 256 | 16 | occupancy | unavailable |  | nsight_compute_not_run |
| gemm_tiled | 256 | 16 | sm_utilization | unavailable |  | nsight_compute_not_run |
| gemm_tiled | 512 | 16 | l2_cache_hit_rate | measured | 89.89 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/gemm_tiled_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:56.215768+00:00 |
| gemm_tiled | 512 | 16 | l2_throughput_pct | measured | 13.78 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/gemm_tiled_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:56.215768+00:00 |
| gemm_tiled | 512 | 16 | memory_throughput_pct | measured | 1.46 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/gemm_tiled_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:56.215768+00:00 |
| gemm_tiled | 512 | 16 | occupancy | measured | 71.72 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/gemm_tiled_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:56.215768+00:00 |
| gemm_tiled | 512 | 16 | sm_utilization | measured | 56.46 | nsight_compute_csv_import|source_tool=ncu|source_file=results/h100-2026-09-22/profiler/gemm_tiled_raw.csv|metric_set=default_profiler_set|import_timestamp=2026-09-22T23:27:56.215768+00:00 |
| memcpy_bandwidth | 1048576 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 1048576 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 1048576 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 1048576 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 1048576 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 4194304 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 4194304 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 4194304 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 4194304 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| memcpy_bandwidth | 4194304 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |
| stencil_1d | 1048576 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| stencil_1d | 1048576 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| stencil_1d | 1048576 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| stencil_1d | 1048576 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| stencil_1d | 1048576 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |
| stencil_1d | 4194304 | 256 | l2_cache_hit_rate | unavailable |  | nsight_compute_not_run |
| stencil_1d | 4194304 | 256 | l2_throughput_pct | unavailable |  | nsight_compute_not_run |
| stencil_1d | 4194304 | 256 | memory_throughput_pct | unavailable |  | nsight_compute_not_run |
| stencil_1d | 4194304 | 256 | occupancy | unavailable |  | nsight_compute_not_run |
| stencil_1d | 4194304 | 256 | sm_utilization | unavailable |  | nsight_compute_not_run |

## Bottleneck Heuristics
| kernel | problem_size | block_size | likely_bottleneck | explanation |
| --- | --- | --- | --- | --- |
| vector_add | 1048576 | 64 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 1048576 | 128 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 1048576 | 256 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 1048576 | 512 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 4194304 | 64 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 4194304 | 128 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 4194304 | 256 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| vector_add | 4194304 | 512 | memory_bound_likely | vector_add: low arithmetic intensity (0.083) suggests memory pressure dominates. |
| reduction | 1048576 | 64 | memory_bound_likely | reduction: low arithmetic intensity (0.246) suggests memory pressure dominates. |
| reduction | 1048576 | 128 | memory_bound_likely | reduction: low arithmetic intensity (0.248) suggests memory pressure dominates. |

## Plot Artifacts
- `plots/runtime_vs_size_vector_reduction.png`
- `plots/runtime_vs_size_gemm.png`
- `plots/effective_gflops_vs_size_gemm.png`
- `plots/effective_bandwidth_vs_size_vector_reduction.png`
- `plots/effective_bandwidth_vs_size_memory_kernels.png`
- `plots/roofline.png`