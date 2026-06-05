#include "kernel_ops.h"

#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

// stencil_1d: a 3-point weighted stencil, out[i] = c0*in[i-1] + c1*in[i] + c2*in[i+1].
//
// Performance behavior: each output reuses neighboring inputs, so this sits between
// the pure-copy (memory-bound) and GEMM (compute-bound) extremes. It illustrates
// halo handling at the array boundaries and modest data reuse: arithmetic intensity
// is higher than a copy but still low, so it remains memory-bound on most GPUs. A
// shared-memory tiled version is a natural next optimization (see Future Work).
namespace {

constexpr float C0 = 0.25f;
constexpr float C1 = 0.50f;
constexpr float C2 = 0.25f;

__global__ void stencil_1d_kernel(const float* in, float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float left = (idx > 0) ? in[idx - 1] : in[idx];
        const float right = (idx + 1 < n) ? in[idx + 1] : in[idx];
        out[idx] = C0 * left + C1 * in[idx] + C2 * right;
    }
}

}  // namespace

BenchmarkRunOutput run_stencil_1d(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify) {
    BenchmarkRunOutput out{};
    out.kernel = "stencil_1d";
    out.problem_size = problem_size;
    out.block_size = block_size;
    out.warmups = warmups;
    out.repeats = repeats;
    out.verify = verify;
    // Counts unique DRAM traffic: read N + write N elements.
    out.bytes_moved = 2ULL * problem_size * sizeof(float);
    // 3 multiplies + 2 adds per element.
    out.flops = 5.0 * static_cast<double>(problem_size);

    std::vector<float> h_in(problem_size, 1.0f);
    std::vector<float> h_out(problem_size, 0.0f);

    float* d_in = nullptr;
    float* d_out = nullptr;
    require_cuda_success(cudaMalloc(&d_in, problem_size * sizeof(float)), "cudaMalloc(d_in)");
    require_cuda_success(cudaMalloc(&d_out, problem_size * sizeof(float)), "cudaMalloc(d_out)");
    require_cuda_success(cudaMemcpy(d_in, h_in.data(), problem_size * sizeof(float), cudaMemcpyHostToDevice), "copy in");

    const int grid = static_cast<int>((problem_size + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        stencil_1d_kernel<<<grid, block_size>>>(d_in, d_out, problem_size);
    }
    require_cuda_success(cudaGetLastError(), "stencil_1d warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    out.runtime_ms_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        require_cuda_success(cudaEventRecord(start), "event start");
        stencil_1d_kernel<<<grid, block_size>>>(d_in, d_out, problem_size);
        require_cuda_success(cudaEventRecord(stop), "event stop");
        require_cuda_success(cudaEventSynchronize(stop), "event sync");
        require_cuda_success(cudaGetLastError(), "stencil_1d timed launch");
        float ms = 0.0f;
        require_cuda_success(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        out.runtime_ms_samples.push_back(ms);
    }

    if (verify) {
        require_cuda_success(cudaMemcpy(h_out.data(), d_out, problem_size * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
        // Input is uniform 1.0, so every interior and boundary output is exactly 1.0.
        out.verification_passed = true;
        for (std::size_t i = 0; i < problem_size; ++i) {
            if (std::fabs(h_out[i] - 1.0f) > 1e-4f) {
                out.verification_passed = false;
                break;
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}
