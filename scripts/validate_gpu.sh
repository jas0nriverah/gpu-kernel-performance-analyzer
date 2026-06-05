#!/usr/bin/env bash
#
# validate_gpu.sh - build the CUDA harness and validate all kernels on a real GPU.
#
# Run this on a CUDA-capable machine (NVIDIA GPU + nvcc + cmake). It is a no-op-safe,
# fail-fast script that:
#   1. confirms the toolchain and GPU are present,
#   2. builds the benchmark binary,
#   3. smoke-tests the newer kernels (memcpy_bandwidth, stencil_1d) with --verify,
#   4. runs the full benchmark sweep, validates artifacts, and generates the report,
#   5. runs the Python test suite.
#
# Usage:
#   bash scripts/validate_gpu.sh
#
# Optional environment variables:
#   BUILD_DIR    cmake build directory      (default: build)
#   OUT_DIR      run output directory       (default: outputs/gpu_validation)
#   PYTHON       python interpreter         (default: python3)
#   PEAK_GFLOPS  real FP32 peak for roofline ceiling   (optional)
#   PEAK_BW_GBPS real DRAM bandwidth peak for roofline (optional)

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
OUT_DIR="${OUT_DIR:-outputs/gpu_validation}"
PYTHON="${PYTHON:-python3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo: $REPO_ROOT"

echo "==> [1/6] Checking toolchain and GPU"
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found on PATH. Install the CUDA Toolkit."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found on PATH."; exit 1; }
nvcc --version | tail -1
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found; continuing, but a GPU is required at runtime."
fi

BIN="$BUILD_DIR/gpu_benchmark"
[ -f "$BIN" ] || BIN="$BUILD_DIR/Release/gpu_benchmark.exe"
[ -f "$BIN" ] || BIN="$BUILD_DIR/gpu_benchmark.exe"

echo "==> [2/6] Building CUDA harness"
cmake -S benchmarks -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --config Release

BIN="$BUILD_DIR/gpu_benchmark"
[ -f "$BIN" ] || BIN="$BUILD_DIR/Release/gpu_benchmark.exe"
[ -f "$BIN" ] || BIN="$BUILD_DIR/gpu_benchmark.exe"
[ -f "$BIN" ] || { echo "ERROR: built binary not found under $BUILD_DIR"; exit 1; }
echo "    binary: $BIN"

echo "==> [3/6] Smoke-testing new kernels directly (with --verify)"
for KERNEL in memcpy_bandwidth stencil_1d; do
  echo "    -- $KERNEL"
  "$BIN" --kernel "$KERNEL" --problem-size 1048576 --block-size 256 --warmups 5 --repeats 10 --verify
done

echo "==> [4/6] Running full benchmark sweep"
"$PYTHON" -m gpu_kernel_analyzer benchmark sweep \
  --binary "$BIN" \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir "$OUT_DIR"

echo "==> [5/6] Validating artifacts and generating analysis/report"
"$PYTHON" -m gpu_kernel_analyzer artifacts validate --run-dir "$OUT_DIR"
ANALYZE_ARGS=(--run-dir "$OUT_DIR")
if [ -n "${PEAK_GFLOPS:-}" ]; then ANALYZE_ARGS+=(--peak-gflops "$PEAK_GFLOPS"); fi
if [ -n "${PEAK_BW_GBPS:-}" ]; then ANALYZE_ARGS+=(--peak-bandwidth-gbps "$PEAK_BW_GBPS"); fi
"$PYTHON" -m gpu_kernel_analyzer analyze full "${ANALYZE_ARGS[@]}"

echo "==> [6/6] Running Python test suite"
"$PYTHON" -m pytest -q

echo
echo "==> DONE. Validation artifacts written to: $OUT_DIR"
echo "    - $OUT_DIR/benchmark_summary.csv"
echo "    - $OUT_DIR/analysis_speedup.csv"
echo "    - $OUT_DIR/REPORT.md"
echo "    - $OUT_DIR/plots/"
