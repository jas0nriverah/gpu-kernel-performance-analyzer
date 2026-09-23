# Cross-GPU validation plan

The current implementation executes CUDA kernels and captures NVIDIA telemetry. H100 timing and power have been exercised locally. Historical A100 timing evidence is retained; fresh A100 power measurements and all other targets remain future validation.

| Target | Execution path | Validation still needed |
| --- | --- | --- |
| A100 | CUDA | Same-revision timing and power rerun; record GA100 sensor semantics |
| H100 | CUDA | Current timing, sustained power, correctness, repeated-trial modeling |
| H200 | CUDA | Same protocol; record exact SKU, memory, clocks, limits |
| L40S | CUDA | Same protocol; verify supported counters and sampling behavior |
| RTX 6000 | CUDA | Record exact model/generation and memory capacity before selecting build architecture |
| AMD MI210 | HIP/ROCm (planned) | Port worker/build and device metadata; add AMD SMI telemetry adapter; verify numerical equivalence |

For every capture retain GPU UUID/model, backend/compiler/driver versions, source/config/binary hashes, actual power limits and clocks, sampling semantics, warmup and duration, input precision, raw timings, and independent trial IDs. Record isolation and thermal conditions. Match workload semantics and input sizes before comparing effective throughput, energy, and numerical correctness. Do not turn a cross-device comparison into a same-device regression gate.

The AMD path is real implementation work, not a switch that makes CUDA code run unchanged. Preserve one worker JSON contract and backend-neutral run identity, then supply CUDA and HIP workers and NVIDIA/AMD telemetry adapters. [HIP portability documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) and [AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/) are the relevant vendor interfaces. Profiler counters must retain their backend-specific meanings; missing metrics must stay unavailable.

Evaluate power models by holding out entire GPU models/devices as well as sessions once those captures exist. Three trials on one H100 establish repeatability only. Do not claim cross-GPU generalization from random rows of the same capture.
