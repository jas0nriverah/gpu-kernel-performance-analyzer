#include "benchmark_runner.h"

#include "cuda_utils.h"
#include "kernel_ops.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliArgs {
    std::string kernel;
    std::size_t problem_size = 0;
    int block_size = 0;
    int warmups = 0;
    int repeats = 0;
    bool verify = false;
};

void print_usage() {
    std::cerr
        << "Usage: gpu_benchmark --kernel <name> --problem-size <N> --block-size <B>"
        << " --warmups <W> --repeats <R> [--verify]\n"
        << "Kernels: vector_add, reduction, gemm_naive, gemm_tiled, memcpy_bandwidth, stencil_1d\n";
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args{};
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--kernel" && i + 1 < argc) {
            args.kernel = argv[++i];
        } else if (token == "--problem-size" && i + 1 < argc) {
            args.problem_size = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (token == "--block-size" && i + 1 < argc) {
            args.block_size = std::stoi(argv[++i]);
        } else if (token == "--warmups" && i + 1 < argc) {
            args.warmups = std::stoi(argv[++i]);
        } else if (token == "--repeats" && i + 1 < argc) {
            args.repeats = std::stoi(argv[++i]);
        } else if (token == "--verify") {
            args.verify = true;
        } else if (token == "--help" || token == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + token);
        }
    }
    if (args.kernel.empty() || args.problem_size == 0 || args.block_size <= 0 || args.warmups < 0 || args.repeats <= 0) {
        throw std::runtime_error("Missing required arguments or invalid values.");
    }
    return args;
}

void write_json_output(const BenchmarkRunOutput& result) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{";
    std::cout << "\"kernel\":\"" << result.kernel << "\",";
    std::cout << "\"problem_size\":" << result.problem_size << ",";
    std::cout << "\"block_size\":" << result.block_size << ",";
    std::cout << "\"warmups\":" << result.warmups << ",";
    std::cout << "\"repeats\":" << result.repeats << ",";
    std::cout << "\"verify\":" << (result.verify ? "true" : "false") << ",";
    std::cout << "\"verification_passed\":" << (result.verification_passed ? "true" : "false") << ",";
    std::cout << "\"bytes_moved\":" << result.bytes_moved << ",";
    std::cout << "\"flops\":" << result.flops << ",";
    std::cout << "\"runtime_ms_samples\":[";
    for (std::size_t i = 0; i < result.runtime_ms_samples.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << result.runtime_ms_samples[i];
    }
    std::cout << "],";
    std::cout << "\"device\":{";
    std::cout << "\"metadata_available\":" << (result.device_info.metadata_available ? "true" : "false") << ",";
    std::cout << "\"name\":\"" << result.device_info.name << "\",";
    std::cout << "\"compute_capability_major\":" << result.device_info.compute_capability_major << ",";
    std::cout << "\"compute_capability_minor\":" << result.device_info.compute_capability_minor << ",";
    std::cout << "\"multiprocessor_count\":" << result.device_info.multiprocessor_count << ",";
    std::cout << "\"total_global_mem_bytes\":" << result.device_info.total_global_mem_bytes;
    std::cout << "}";
    std::cout << "}" << std::endl;
}

}  // namespace

int run_benchmark_cli(int argc, char** argv) {
    try {
        const CliArgs args = parse_args(argc, argv);
        BenchmarkRunOutput result{};
        if (args.kernel == "vector_add") {
            result = run_vector_add(args.problem_size, args.block_size, args.warmups, args.repeats, args.verify);
        } else if (args.kernel == "reduction") {
            result = run_reduction(args.problem_size, args.block_size, args.warmups, args.repeats, args.verify);
        } else if (args.kernel == "gemm_naive") {
            result = run_gemm_naive(args.problem_size, args.block_size, args.warmups, args.repeats, args.verify);
        } else if (args.kernel == "gemm_tiled") {
            result = run_gemm_tiled(args.problem_size, args.block_size, args.warmups, args.repeats, args.verify);
        } else if (args.kernel == "memcpy_bandwidth") {
            result = run_memcpy_bandwidth(args.problem_size, args.block_size, args.warmups, args.repeats, args.verify);
        } else if (args.kernel == "stencil_1d") {
            result = run_stencil_1d(args.problem_size, args.block_size, args.warmups, args.repeats, args.verify);
        } else {
            throw std::runtime_error("Unsupported kernel: " + args.kernel);
        }
        result.device_info = query_device_info();
        write_json_output(result);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        print_usage();
        return 1;
    }
}
