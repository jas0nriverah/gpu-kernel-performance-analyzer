#pragma once

#include "benchmark_types.h"

DeviceInfo query_device_info();
void require_cuda_success(int code, const char* operation);

#include <functional>

// Optional sustained workload for board-power sampling, separate from event timings.
void set_sustain_seconds(double seconds);
void run_sustained(const std::function<void()>& launch);
