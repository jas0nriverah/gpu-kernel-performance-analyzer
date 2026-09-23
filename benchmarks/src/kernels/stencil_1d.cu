#include "kernel_ops.h"

#include "cuda_utils.h"
#include "validation.h"

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
    const std::size_t bytes = checked_bytes(problem_size, sizeof(float));
    const int grid = validate_1d_launch(problem_size, block_size);
    if (problem_size > static_cast<std::size_t>(-1) / (2 * sizeof(float))) throw std::runtime_error("byte count overflows");
    out.bytes_moved = 2ULL * problem_size * sizeof(float);
    // 3 multiplies + 2 adds per element.
    out.flops = 5.0 * static_cast<double>(problem_size);

    std::vector<float> h_in(problem_size);
    for (std::size_t i = 0; i < problem_size; ++i) h_in[i] = input_value(i, 4);
    std::vector<float> h_out(problem_size, 0.0f);

    float* d_in = nullptr;
    float* d_out = nullptr;
    require_cuda_success(cudaMalloc(&d_in, bytes), "cudaMalloc(d_in)");
    require_cuda_success(cudaMalloc(&d_out, bytes), "cudaMalloc(d_out)");
    require_cuda_success(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), "copy in");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        stencil_1d_kernel<<<grid, block_size>>>(d_in, d_out, problem_size);
    }
    require_cuda_success(cudaGetLastError(), "stencil_1d warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    run_sustained([&]() { stencil_1d_kernel<<<grid, block_size>>>(d_in, d_out, problem_size); });

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
        require_cuda_success(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost), "copy out");
        out.verification_passed = true;
        for (std::size_t i = 0; i < problem_size; ++i) {
            const double left = (i > 0) ? h_in[i - 1] : h_in[i];
            const double right = (i + 1 < problem_size) ? h_in[i + 1] : h_in[i];
            const double expected = 0.25 * left + 0.50 * h_in[i] + 0.25 * right;
            if (!close_enough(h_out[i], expected)) {
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
