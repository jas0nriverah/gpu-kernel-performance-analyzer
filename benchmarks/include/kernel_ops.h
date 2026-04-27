#pragma once

#include "benchmark_types.h"

BenchmarkRunOutput run_vector_add(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify);
BenchmarkRunOutput run_reduction(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify);
BenchmarkRunOutput run_gemm_naive(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify);
BenchmarkRunOutput run_gemm_tiled(std::size_t problem_size, int block_size, int warmups, int repeats, bool verify);
