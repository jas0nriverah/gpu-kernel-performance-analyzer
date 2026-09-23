#include "kernel_ops.h"

#include "cuda_utils.h"
#include "validation.h"

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

    const unsigned int grid_dim = validate_gemm_launch(problem_size, block_size);
    const int n = checked_problem_int(problem_size);
    if (problem_size > static_cast<std::size_t>(-1) / problem_size) throw std::runtime_error("GEMM element count overflows");
    const std::size_t matrix_elems = problem_size * problem_size;
    const std::size_t matrix_bytes = checked_bytes(matrix_elems, sizeof(float));
    out.bytes_moved = 3ULL * matrix_bytes;
    out.flops = 2.0 * static_cast<double>(problem_size) * static_cast<double>(problem_size) * static_cast<double>(problem_size);

    std::vector<float> h_a(matrix_elems), h_b(matrix_elems);
    // Exhaustive arbitrary seeded matrices through N=128 exercise every dot
    // product against the CPU reference. Larger cases use nonuniform rank-one
    // factors so every output still has an O(1) closed-form reference and the
    // CPU verifier stays O(N^2), at the cost of reduced large-N data diversity.
    for (std::size_t r = 0; r < problem_size; ++r) for (std::size_t k = 0; k < problem_size; ++k) {
        h_a[r * problem_size + k] = problem_size <= 128 ? input_value(r * problem_size + k, 23) : row_factor(r) * gemm_x(k);
        h_b[k * problem_size + r] = problem_size <= 128 ? input_value(k * problem_size + r, 37) : gemm_y(k) * col_factor(r);
    }
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
    const dim3 grid(grid_dim, grid_dim);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    require_cuda_success(cudaGetLastError(), "gemm_naive warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    run_sustained([&]() { gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, n); });

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
        out.verification_passed = true;
        double shared_dot = 0.0;
        if (problem_size > 128) for (std::size_t k = 0; k < problem_size; ++k) shared_dot += static_cast<double>(gemm_x(k)) * gemm_y(k);
        for (std::size_t r = 0; r < problem_size && out.verification_passed; ++r) for (std::size_t c = 0; c < problem_size; ++c) {
            double expected = 0.0;
            if (problem_size <= 128) for (std::size_t k = 0; k < problem_size; ++k)
                expected += static_cast<double>(h_a[r * problem_size + k]) * h_b[k * problem_size + c];
            else expected = static_cast<double>(row_factor(r)) * col_factor(c) * shared_dot;
            if (!close_enough(h_c[r * problem_size + c], expected, 1e-3, 5e-4)) {
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
