#include "kernel_ops.h"

#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

}  // namespace

BenchmarkRunOutput run_vector_add(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify) {
    BenchmarkRunOutput out{};
    out.kernel = "vector_add";
    out.problem_size = problem_size;
    out.block_size = block_size;
    out.warmups = warmups;
    out.repeats = repeats;
    out.verify = verify;
    out.bytes_moved = 3ULL * problem_size * sizeof(float);
    out.flops = static_cast<double>(problem_size);

    std::vector<float> h_a(problem_size, 1.5f);
    std::vector<float> h_b(problem_size, 2.5f);
    std::vector<float> h_c(problem_size, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    require_cuda_success(cudaMalloc(&d_a, problem_size * sizeof(float)), "cudaMalloc(d_a)");
    require_cuda_success(cudaMalloc(&d_b, problem_size * sizeof(float)), "cudaMalloc(d_b)");
    require_cuda_success(cudaMalloc(&d_c, problem_size * sizeof(float)), "cudaMalloc(d_c)");

    require_cuda_success(cudaMemcpy(d_a, h_a.data(), problem_size * sizeof(float), cudaMemcpyHostToDevice), "copy a");
    require_cuda_success(cudaMemcpy(d_b, h_b.data(), problem_size * sizeof(float), cudaMemcpyHostToDevice), "copy b");

    const int grid = static_cast<int>((problem_size + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        vector_add_kernel<<<grid, block_size>>>(d_a, d_b, d_c, problem_size);
    }
    require_cuda_success(cudaGetLastError(), "vector_add warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    out.runtime_ms_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        require_cuda_success(cudaEventRecord(start), "event start");
        vector_add_kernel<<<grid, block_size>>>(d_a, d_b, d_c, problem_size);
        require_cuda_success(cudaEventRecord(stop), "event stop");
        require_cuda_success(cudaEventSynchronize(stop), "event sync");
        require_cuda_success(cudaGetLastError(), "vector_add timed launch");
        float ms = 0.0f;
        require_cuda_success(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        out.runtime_ms_samples.push_back(ms);
    }

    if (verify) {
        require_cuda_success(cudaMemcpy(h_c.data(), d_c, problem_size * sizeof(float), cudaMemcpyDeviceToHost), "copy c");
        for (std::size_t i = 0; i < problem_size; ++i) {
            const float expected = h_a[i] + h_b[i];
            if (std::fabs(h_c[i] - expected) > 1e-4f) {
                out.verification_passed = false;
                break;
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return out;
}
