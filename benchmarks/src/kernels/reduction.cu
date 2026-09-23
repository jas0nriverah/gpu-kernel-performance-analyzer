#include "kernel_ops.h"

#include "cuda_utils.h"
#include "validation.h"

#include <cuda_runtime.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

__global__ void reduce_sum_block(const float* input, float* block_sums, std::size_t n) {
    extern __shared__ float shared[];
    const unsigned int tid = threadIdx.x;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;

    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0];
    }
}

}  // namespace

BenchmarkRunOutput run_reduction(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify) {
    BenchmarkRunOutput out{};
    out.kernel = "reduction";
    out.problem_size = problem_size;
    out.block_size = block_size;
    out.warmups = warmups;
    out.repeats = repeats;
    out.verify = verify;

    if (block_size <= 0 || (block_size & (block_size - 1)) != 0) throw std::runtime_error("reduction block_size must be a power of two");
    const int grid = validate_1d_launch(problem_size, block_size);
    const std::size_t input_bytes = checked_bytes(problem_size, sizeof(float));
    const std::size_t partial_bytes = checked_bytes(static_cast<std::size_t>(grid), sizeof(float));
    out.bytes_moved = (problem_size + static_cast<std::size_t>(grid)) * sizeof(float);
    out.flops = (problem_size > 0) ? static_cast<double>(problem_size - 1) : 0.0;

    std::vector<float> h_in(problem_size);
    for (std::size_t i = 0; i < problem_size; ++i) h_in[i] = input_value(i, 31);
    std::vector<float> h_block(static_cast<std::size_t>(grid), 0.0f);

    float* d_in = nullptr;
    float* d_block = nullptr;
    require_cuda_success(cudaMalloc(&d_in, input_bytes), "cudaMalloc(d_in)");
    require_cuda_success(cudaMalloc(&d_block, partial_bytes), "cudaMalloc(d_block)");
    require_cuda_success(cudaMemcpy(d_in, h_in.data(), input_bytes, cudaMemcpyHostToDevice), "copy reduction input");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    require_cuda_success(cudaEventCreate(&start), "cudaEventCreate(start)");
    require_cuda_success(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const std::size_t shared_mem = static_cast<std::size_t>(block_size) * sizeof(float);

    for (int i = 0; i < warmups; ++i) {
        reduce_sum_block<<<grid, block_size, shared_mem>>>(d_in, d_block, problem_size);
    }
    require_cuda_success(cudaGetLastError(), "reduction warmup launch");
    require_cuda_success(cudaDeviceSynchronize(), "warmup sync");

    run_sustained([&]() { reduce_sum_block<<<grid, block_size, shared_mem>>>(d_in, d_block, problem_size); });

    out.runtime_ms_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        require_cuda_success(cudaEventRecord(start), "event start");
        reduce_sum_block<<<grid, block_size, shared_mem>>>(d_in, d_block, problem_size);
        require_cuda_success(cudaEventRecord(stop), "event stop");
        require_cuda_success(cudaEventSynchronize(stop), "event sync");
        require_cuda_success(cudaGetLastError(), "reduction timed launch");
        float ms = 0.0f;
        require_cuda_success(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        out.runtime_ms_samples.push_back(ms);
    }

    if (verify) {
        require_cuda_success(
            cudaMemcpy(h_block.data(), d_block, partial_bytes, cudaMemcpyDeviceToHost),
            "copy block sums"
        );
        out.verification_passed = true;
        for (int b = 0; b < grid && out.verification_passed; ++b) {
            const std::size_t begin = static_cast<std::size_t>(b) * block_size;
            const std::size_t end = (begin + static_cast<std::size_t>(block_size) < problem_size)
                ? begin + static_cast<std::size_t>(block_size) : problem_size;
            double expected = 0.0;
            for (std::size_t i = begin; i < end; ++i) expected += h_in[i];
            out.verification_passed = close_enough(h_block[static_cast<std::size_t>(b)], expected, 2e-4, 2e-5);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_block);
    return out;
}
