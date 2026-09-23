#include "kernel_ops.h"

#include "cuda_utils.h"
#include "validation.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

// memcpy_bandwidth: a pure streaming copy (out[i] = in[i]).
//
// Performance behavior: this kernel does no arithmetic, so it is the canonical
// memory-bound case. Its effective bandwidth (2 * N * sizeof(float) bytes moved
// over the measured runtime) is the closest single-kernel proxy for the device's
// achievable DRAM streaming bandwidth, and serves as the practical "memory roof"
// reference point for the other kernels.
namespace {

__global__ void copy_kernel(const float* in, float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

}  // namespace

BenchmarkRunOutput run_memcpy_bandwidth(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify) {
    BenchmarkRunOutput out{};
    out.kernel = "memcpy_bandwidth";
    out.problem_size = problem_size;
    out.block_size = block_size;
    out.warmups = warmups;
    out.repeats = repeats;
    out.verify = verify;
    // One read + one write per element.
    const std::size_t bytes = checked_bytes(problem_size, sizeof(float));
    const int grid = validate_1d_launch(problem_size, block_size);
    if (problem_size > static_cast<std::size_t>(-1) / (2 * sizeof(float))) throw std::runtime_error("byte count overflows");
    out.bytes_moved = 2ULL * problem_size * sizeof(float);
    out.flops = 0.0;

    std::vector<float> h_in(problem_size);
    for (std::size_t i = 0; i < problem_size; ++i) h_in[i] = input_value(i, 5);
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
        copy_kernel<<<grid, block_size>>>(d_in, d_out, problem_size);
    }
    require_cuda_success(cudaGetLastError(), "memcpy_bandwidth warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    run_sustained([&]() { copy_kernel<<<grid, block_size>>>(d_in, d_out, problem_size); });

    out.runtime_ms_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        require_cuda_success(cudaEventRecord(start), "event start");
        copy_kernel<<<grid, block_size>>>(d_in, d_out, problem_size);
        require_cuda_success(cudaEventRecord(stop), "event stop");
        require_cuda_success(cudaEventSynchronize(stop), "event sync");
        require_cuda_success(cudaGetLastError(), "memcpy_bandwidth timed launch");
        float ms = 0.0f;
        require_cuda_success(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        out.runtime_ms_samples.push_back(ms);
    }

    if (verify) {
        require_cuda_success(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost), "copy out");
        out.verification_passed = true;
        for (std::size_t i = 0; i < problem_size; ++i) {
            if (!close_enough(h_out[i], h_in[i])) {
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
