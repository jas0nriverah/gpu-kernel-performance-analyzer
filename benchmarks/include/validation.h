#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Deterministic, bounded values make verification reproducible while exercising
// indexing, tails, signs, and nonuniform arithmetic.
inline std::uint64_t validation_hash(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
inline float input_value(std::size_t i, unsigned salt = 0) {
    const int v = static_cast<int>(validation_hash(static_cast<std::uint64_t>(i) ^
                                                   (static_cast<std::uint64_t>(salt) << 48)) & 0x1fu) - 16;
    return static_cast<float>(v) / 16.0f;
}
inline float row_factor(std::size_t i) { return input_value(i, 3) + 0.125f; }
inline float col_factor(std::size_t i) { return input_value(i, 11) - 0.1875f; }
inline float gemm_x(std::size_t i) { return input_value(i, 7); }
inline float gemm_y(std::size_t i) { return input_value(i, 19); }

inline bool close_enough(float got, double expected, double abs_tol = 2e-4,
                         double rel_tol = 2e-5) {
    return std::isfinite(got) && std::isfinite(expected) &&
           std::fabs(static_cast<double>(got) - expected) <=
               abs_tol + rel_tol * std::fabs(expected);
}

inline std::size_t checked_bytes(std::size_t count, std::size_t element_bytes) {
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / element_bytes)
        throw std::runtime_error("problem size is zero or byte size overflows");
    return count * element_bytes;
}

inline int checked_problem_int(std::size_t n) {
    if (n == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("problem size exceeds supported kernel indexing");
    return static_cast<int>(n);
}
