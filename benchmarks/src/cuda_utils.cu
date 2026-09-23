#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <limits>

#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <string>

void require_cuda_success(int code, const char* operation) {
    if (code != cudaSuccess) {
        const char* error_name = cudaGetErrorString(static_cast<cudaError_t>(code));
        throw std::runtime_error(std::string("CUDA failure during ") + operation + ": " + error_name);
    }
}

namespace {
cudaDeviceProp current_properties() {
    int device = 0;
    require_cuda_success(cudaGetDevice(&device), "cudaGetDevice for launch validation");
    cudaDeviceProp prop{};
    require_cuda_success(cudaGetDeviceProperties(&prop, device), "device properties for launch validation");
    return prop;
}
}

int validate_1d_launch(std::size_t n, int block_size) {
    if (n == 0 || block_size <= 0) throw std::runtime_error("problem size and block size must be positive");
    const cudaDeviceProp prop = current_properties();
    if (block_size > prop.maxThreadsPerBlock || block_size > prop.maxThreadsDim[0])
        throw std::runtime_error("1D block size exceeds device launch limits");
    const std::size_t grid = n / static_cast<std::size_t>(block_size) +
                             (n % static_cast<std::size_t>(block_size) != 0);
    if (grid > static_cast<std::size_t>(prop.maxGridSize[0]) ||
        grid > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("1D grid dimension exceeds device launch limits");
    return static_cast<int>(grid);
}

unsigned int validate_gemm_launch(std::size_t n, int block_size) {
    if (n == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || block_size <= 0)
        throw std::runtime_error("GEMM size or block size exceeds supported kernel indexing");
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) / n)
        throw std::runtime_error("GEMM linear index would overflow signed kernel indexing");
    const cudaDeviceProp prop = current_properties();
    if (block_size > prop.maxThreadsDim[0] || block_size > prop.maxThreadsDim[1] ||
        static_cast<long long>(block_size) * block_size > prop.maxThreadsPerBlock)
        throw std::runtime_error("GEMM block exceeds device launch limits");
    const std::size_t grid = n / static_cast<std::size_t>(block_size) +
                             (n % static_cast<std::size_t>(block_size) != 0);
    if (grid > static_cast<std::size_t>(prop.maxGridSize[0]) ||
        grid > static_cast<std::size_t>(prop.maxGridSize[1]))
        throw std::runtime_error("GEMM grid dimension exceeds device launch limits");
    return static_cast<unsigned int>(grid);
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
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < sizeof(prop.uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
        uuid << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    info.uuid = "GPU-" + uuid.str();
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
