# Kernel correctness and launch validation

Each verified run uses repeatable seeded, nonuniform single-precision inputs. Vector addition and copy compare every output; the stencil compares every point against a CPU stencil with matching edge clamping; reduction checks each GPU block partial separately against its CPU sum. Comparisons require finite values and use absolute plus relative tolerances.

GEMM checks every output. Through `N=128`, both inputs are arbitrary seeded matrices and the CPU computes every dot product directly. Larger matrices use distinct signed row and column factors with separate signed reduction vectors. That structure gives each output a closed-form reference and keeps the CPU verification cost at O(N²), matching the number of outputs. Large cases therefore exercise every output and indexing path, while offering less input diversity than arbitrary dense matrices. GPU accumulation can differ slightly from the double-precision CPU oracle, so GEMM uses a scale-aware tolerance.

Scenario expansion rejects non-integer, zero, or unsafe sizes; blocks over 1024 threads; non-power-of-two reduction blocks; GEMM signed-index overflow; and launch grids beyond conservative CUDA bounds. Native runners repeat the checks against the active CUDA device properties before allocating buffers. GEMM also checks `N*N` against the signed integer indexing used by the kernels. Use `gpu_benchmark --device-info` to query the active device without launching a benchmark; its JSON includes `device.uuid` in NVIDIA `GPU-...` format.

[H100 validation artifacts](../results/h100-correctness-expansion-2026-09-22/) retain worker output, rejected-input logs, build commands, and source/binary hashes.
