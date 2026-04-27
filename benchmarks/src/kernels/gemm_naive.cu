#include "kernel_ops.h"

#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

__global__ void gemm_naive_kernel(const float* a, const float* b, float* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }
}

}  // namespace

BenchmarkRunOutput run_gemm_naive(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify) {
    BenchmarkRunOutput out{};
    out.kernel = "gemm_naive";
    out.problem_size = problem_size;
    out.block_size = block_size;
    out.warmups = warmups;
    out.repeats = repeats;
    out.verify = verify;

    const int n = static_cast<int>(problem_size);
    const std::size_t matrix_elems = problem_size * problem_size;
    const std::size_t matrix_bytes = matrix_elems * sizeof(float);
    out.bytes_moved = 3ULL * matrix_bytes;
    out.flops = 2.0 * static_cast<double>(problem_size) * static_cast<double>(problem_size) * static_cast<double>(problem_size);

    std::vector<float> h_a(matrix_elems, 1.0f);
    std::vector<float> h_b(matrix_elems, 1.0f);
    std::vector<float> h_c(matrix_elems, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    require_cuda_success(cudaMalloc(&d_a, matrix_bytes), "cudaMalloc(d_a)");
    require_cuda_success(cudaMalloc(&d_b, matrix_bytes), "cudaMalloc(d_b)");
    require_cuda_success(cudaMalloc(&d_c, matrix_bytes), "cudaMalloc(d_c)");
    require_cuda_success(cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice), "copy A");
    require_cuda_success(cudaMemcpy(d_b, h_b.data(), matrix_bytes, cudaMemcpyHostToDevice), "copy B");

    const dim3 block(static_cast<unsigned int>(block_size), static_cast<unsigned int>(block_size));
    const dim3 grid(
        static_cast<unsigned int>((n + block_size - 1) / block_size),
        static_cast<unsigned int>((n + block_size - 1) / block_size)
    );

    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    require_cuda_success(cudaGetLastError(), "gemm_naive warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    out.runtime_ms_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        require_cuda_success(cudaEventRecord(start), "event start");
        gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        require_cuda_success(cudaEventRecord(stop), "event stop");
        require_cuda_success(cudaEventSynchronize(stop), "event sync");
        require_cuda_success(cudaGetLastError(), "gemm_naive timed launch");
        float ms = 0.0f;
        require_cuda_success(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        out.runtime_ms_samples.push_back(ms);
    }

    if (verify) {
        require_cuda_success(cudaMemcpy(h_c.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost), "copy C");
        const float expected = static_cast<float>(n);
        out.verification_passed = true;
        for (int i = 0; i < 8 && i < n; ++i) {
            const float value = h_c[static_cast<std::size_t>(i) * n + i];
            if (std::fabs(value - expected) > 1e-2f) {
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
