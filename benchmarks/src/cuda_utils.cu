#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

void require_cuda_success(int code, const char* operation) {
    if (code != cudaSuccess) {
        const char* error_name = cudaGetErrorString(static_cast<cudaError_t>(code));
        throw std::runtime_error(std::string("CUDA failure during ") + operation + ": " + error_name);
    }
}

DeviceInfo query_device_info() {
    DeviceInfo info{};
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return info;
    }

    int current_device = 0;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        current_device = 0;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, current_device) != cudaSuccess) {
        return info;
    }

    info.name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.total_global_mem_bytes = static_cast<std::size_t>(prop.totalGlobalMem);
    info.metadata_available = true;
    return info;
}
