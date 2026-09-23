#include "kernel_ops.h"

#include "cuda_utils.h"
#include "validation.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    const int tile_count = (n + TILE - 1) / TILE;
    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * TILE + threadIdx.x;
        const int b_row = tile * TILE + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}

}  // namespace

BenchmarkRunOutput run_gemm_tiled(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify) {
    if (block_size != TILE) {
        throw std::runtime_error("gemm_tiled currently requires block_size=16 for MVP.");
    }

    BenchmarkRunOutput out{};
    out.kernel = "gemm_tiled";
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

    const dim3 block(TILE, TILE);
    const dim3 grid(grid_dim, grid_dim);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    require_cuda_success(cudaGetLastError(), "gemm_tiled warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    run_sustained([&]() { gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n); });

    out.runtime_ms_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        require_cuda_success(cudaEventRecord(start), "event start");
        gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        require_cuda_success(cudaEventRecord(stop), "event stop");
        require_cuda_success(cudaEventSynchronize(stop), "event sync");
        require_cuda_success(cudaGetLastError(), "gemm_tiled timed launch");
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
