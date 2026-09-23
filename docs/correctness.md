# Kernel correctness and launch validation

Each verified run now uses repeatable seeded, nonuniform single-precision inputs. Vector addition and copy compare every output; the stencil compares every point against a CPU stencil with matching edge clamping; reduction checks each GPU block partial separately against its CPU sum. Comparisons require finite values and use absolute plus relative tolerances.

GEMM checks every output. Through `N=128`, both inputs are arbitrary seeded matrices and the CPU computes every dot product directly. Larger matrices use distinct signed row and column factors with separate signed reduction vectors. That structure gives each output a closed-form reference and keeps the CPU verification cost at O(N²), matching the number of outputs. Large cases therefore exercise every output and indexing path, while offering less input diversity than arbitrary dense matrices. GPU accumulation can differ slightly from the double-precision CPU oracle, so GEMM uses a scale-aware tolerance.

Scenario expansion rejects non-integer, zero, or unsafe sizes; blocks over 1024 threads; non-power-of-two reduction blocks; GEMM signed-index overflow; and launch grids beyond conservative CUDA bounds. Native runners repeat the checks against the active CUDA device properties before allocating buffers. GEMM also checks `N*N` against the signed integer indexing used by the kernels. Use `gpu_benchmark --device-info` to query the active device without launching a benchmark; its JSON includes `device.uuid` in NVIDIA `GPU-...` format.

## Validation record

The run artifacts are retained under [`results/h100-correctness-expansion-2026-09-22`](../results/h100-correctness-expansion-2026-09-22/), including worker JSON, rejected-request logs, build command, and source/binary hashes. The binary was built for the native H100 architecture with:

```bash
cmake -S benchmarks -B /tmp/gpu-expansion-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build /tmp/gpu-expansion-build -j2
```

On NVIDIA H100 80GB HBM3 (`GPU-91e942dc-edbb-7f2a-9cea-167356bbd86a`), all 24 scenarios in `configs/benchmark_scenarios.yaml` and all 18 cases in `configs/correctness_edges.json` completed with verification passing. Both 2048 × 2048 GEMMs passed full-output checks. Vector add, copy, stencil, and reduction each passed at 67,108,864 elements with short timing runs. Direct worker invocations rejected zero sizes, reduction block size 96, block size 2048, a grid beyond device limits, and a GEMM size whose linear index would overflow.

The UUID from `--device-info` matched `nvidia-smi`. CUDA validation used `CMAKE_CUDA_ARCHITECTURES=90`; the compiler's default SM_75 target failed to launch on this H100 because its PTX was rejected by the installed driver.
