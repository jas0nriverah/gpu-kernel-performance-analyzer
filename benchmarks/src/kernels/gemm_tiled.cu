#include "kernel_ops.h"

#include "cuda_utils.h"

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

    const dim3 block(TILE, TILE);
    const dim3 grid(
        static_cast<unsigned int>((n + TILE - 1) / TILE),
        static_cast<unsigned int>((n + TILE - 1) / TILE)
    );

    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmups; ++i) {
        gemm_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    require_cuda_success(cudaGetLastError(), "gemm_tiled warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

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
