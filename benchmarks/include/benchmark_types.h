#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct DeviceInfo {
    std::string name = "unavailable";
    int compute_capability_major = -1;
    int compute_capability_minor = -1;
    int multiprocessor_count = -1;
    std::size_t total_global_mem_bytes = 0;
    bool metadata_available = false;
};

struct BenchmarkRunOutput {
    std::string kernel;
    std::size_t problem_size = 0;
    int block_size = 0;
    int warmups = 0;
    int repeats = 0;
    bool verify = false;

    std::vector<float> runtime_ms_samples;
    std::size_t bytes_moved = 0;
    double flops = 0.0;
    bool verification_passed = true;

    DeviceInfo device_info;
};
