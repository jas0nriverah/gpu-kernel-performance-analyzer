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

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {
double sustain_seconds = 0.0;
double epoch_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
}

void set_sustain_seconds(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0 || seconds > 3600) {
        throw std::runtime_error("sustain-seconds must be finite and between 0 and 3600");
    }
    sustain_seconds = seconds;
}

void run_sustained(const std::function<void()>& launch) {
    if (sustain_seconds == 0.0) return;
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    const double start_epoch = epoch_seconds();
    unsigned long long launches = 0;
    double elapsed = 0;
    do {
        for (int batch = 0; batch < 100; ++batch) {
            launch();
            ++launches;
        }
        require_cuda_success(cudaGetLastError(), "sustained launch");
        require_cuda_success(cudaDeviceSynchronize(), "sustained sync");
        elapsed = std::chrono::duration<double>(clock::now() - start).count();
    } while (elapsed < sustain_seconds);
    const double end_epoch = epoch_seconds();
    std::cerr << std::setprecision(17) << "POWER_WINDOW {\"start_epoch_s\":" << start_epoch
              << ",\"end_epoch_s\":" << end_epoch << ",\"duration_s\":" << elapsed
              << ",\"launches\":" << launches << "}\n";
}
