# Metrics Policy (MVP)

This project enforces explicit metric provenance for every reported value.

## Default metrics

- `runtime_ms` - `measured` from CUDA event timing.
- `effective_bandwidth_GBps` - `derived_estimate` from bytes moved and measured runtime.
- `effective_GFLOPs` - `derived_estimate` from FLOP estimate and measured runtime.
- `arithmetic_intensity` - `derived_estimate` from FLOPs / bytes moved.
- `device_metadata` - `measured` from CUDA runtime API when available; otherwise `unavailable`.

## Profiler-dependent metrics

- `occupancy`
- `sm_utilization`
- `memory_throughput_pct`
- `l2_throughput_pct`
- `l2_cache_hit_rate`

These are `unavailable` in MVP unless Nsight Compute data is explicitly collected and parsed.
Use:

`python -m gpu_kernel_analyzer profile ncu-normalize --raw-csv <raw_ncu_csv> --out-csv <normalized_csv> --kernel <kernel> --problem-size <size> --block-size <block> --metric-set <set_name>`

`python -m gpu_kernel_analyzer profile ncu-import --run-dir <run_dir> --source-tool ncu --source-file <raw_artifact> --metric-set <set_name> --ncu-csv <normalized_csv>`

Import requirements:

- CSV columns must include: `kernel,problem_size,block_size,metric_name,metric_value`
- each imported row must map to an exact benchmark scenario (`kernel + problem_size + block_size`)
- provenance metadata (`source_tool`, `source_file`, `metric_set`, `import_timestamp`) is required
- imported profiler metrics are scenario-specific; unprofiled scenarios remain unavailable
- Nsight profiling runtime overhead is not used for benchmark `runtime_ms` claims (CUDA events remain source of truth)

## Enforcement

- Artifact validation fails if profiler-dependent metrics are marked as measured without profiler integration.
- Missing or malformed required artifact files fail validation.
- Profiler metrics marked measured without required provenance metadata are rejected by validation.
