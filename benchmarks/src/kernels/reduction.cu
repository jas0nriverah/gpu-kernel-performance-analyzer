#include "kernel_ops.h"

#include "cuda_utils.h"

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

    const int grid = static_cast<int>((problem_size + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
    out.bytes_moved = (problem_size + static_cast<std::size_t>(grid)) * sizeof(float);
    out.flops = (problem_size > 0) ? static_cast<double>(problem_size - 1) : 0.0;

    std::vector<float> h_in(problem_size, 1.0f);
    std::vector<float> h_block(static_cast<std::size_t>(grid), 0.0f);

    float* d_in = nullptr;
    float* d_block = nullptr;
    require_cuda_success(cudaMalloc(&d_in, problem_size * sizeof(float)), "cudaMalloc(d_in)");
    require_cuda_success(cudaMalloc(&d_block, static_cast<std::size_t>(grid) * sizeof(float)), "cudaMalloc(d_block)");
    require_cuda_success(cudaMemcpy(d_in, h_in.data(), problem_size * sizeof(float), cudaMemcpyHostToDevice), "copy reduction input");

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
            cudaMemcpy(h_block.data(), d_block, static_cast<std::size_t>(grid) * sizeof(float), cudaMemcpyDeviceToHost),
            "copy block sums"
        );
        const float gpu_sum = std::accumulate(h_block.begin(), h_block.end(), 0.0f);
        const float cpu_sum = static_cast<float>(problem_size);
        out.verification_passed = std::fabs(gpu_sum - cpu_sum) <= 1e-2f * cpu_sum;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_block);
    return out;
}
