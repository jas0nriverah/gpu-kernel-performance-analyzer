#pragma once

#include "benchmark_types.h"

DeviceInfo query_device_info();
void require_cuda_success(int code, const char* operation);
