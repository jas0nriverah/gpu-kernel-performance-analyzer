# Interview Defense Notes

## What is measured vs estimated

- Measured:
  - kernel runtime with CUDA events (`runtime_ms`)
  - device metadata from CUDA runtime API (when available)
- Derived estimate:
  - effective bandwidth
  - effective GFLOPs
  - arithmetic intensity
- Unavailable by default:
  - occupancy
  - SM utilization
  - L2 cache hit rate

## Why this is credible

- Raw timing samples are persisted, not just aggregated metrics.
- Artifact schema validation enforces required files and columns.
- Provenance table explicitly labels every metric status.
- Profiler metrics cannot be marked measured unless Nsight data is imported.

## Known limitations

- No GPU CI in this repository by default.
- Kernel set is intentionally small for MVP.
- Derived throughput estimates depend on operation/byte count assumptions.

## Extension path

- Add optional Nsight Compute runner invocation.
- Expand kernel set and optimization variants.
- Add more sophisticated bottleneck analysis and roofline visualizations.
