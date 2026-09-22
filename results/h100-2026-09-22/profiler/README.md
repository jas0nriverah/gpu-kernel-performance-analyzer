# H100 hardware counter captures

Each raw CSV captures one launch after warmup, using Nsight Compute's default replay/cache/clock controls. The final vector capture and GEMM capture ran sequentially after timing sweeps and edge checks had finished. The raw stdout includes heavily perturbed CUDA-event durations; none of those durations are used in benchmark summaries.

Capture commands (from repository root):

```bash
METRICS=sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct
ncu --csv --page raw --launch-skip 10 --launch-count 1 --metrics "$METRICS" \
  build/gpu_benchmark --kernel vector_add --problem-size 4194304 --block-size 256 \
  --warmups 10 --repeats 30 --verify > vector_add_raw.csv
ncu --csv --page raw --launch-skip 5 --launch-count 1 --metrics "$METRICS" \
  build/gpu_benchmark --kernel gemm_tiled --problem-size 512 --block-size 16 \
  --warmups 5 --repeats 15 --verify > gemm_tiled_raw.csv
```

The normalized files use the project's `default_profiler_set`; they were imported into `../baseline/` using `profile ncu-import`. The metric set recognizes additional optional source names, but only the five requested counters above were captured. `version.txt` records the profiler version. `stderr.txt` is empty because both captures succeeded.
