cmake_minimum_required(VERSION 3.21)
project(gpu_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(gpu_benchmark
    src/main.cu
    src/benchmark_runner.cu
    src/cuda_utils.cu
    src/kernels/vector_add.cu
    src/kernels/reduction.cu
    src/kernels/gemm_naive.cu
    src/kernels/gemm_tiled.cu
)

target_include_directories(gpu_benchmark PRIVATE include)

if(MSVC)
  target_compile_options(gpu_benchmark PRIVATE /W3)
else()
  target_compile_options(gpu_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()
