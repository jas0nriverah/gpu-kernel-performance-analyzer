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
#   BUILD_DIR        cmake build directory      (default: build)
#   OUT_DIR          run output directory       (default: outputs/gpu_validation)
#   PYTHON           python interpreter         (default: python3)
#   SKIP_PIP_INSTALL set to 1 to skip the editable install step (default: unset)
#   CUDAToolkit_ROOT CUDA toolkit root if CMake cannot find nvcc   (optional)
#   PEAK_GFLOPS      real FP32 peak for roofline ceiling           (optional)
#   PEAK_BW_GBPS     real DRAM bandwidth peak for roofline         (optional)

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
OUT_DIR="${OUT_DIR:-outputs/gpu_validation}"
PYTHON="${PYTHON:-python3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo: $REPO_ROOT"

echo "==> [1/7] Checking toolchain and GPU"
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found on PATH. Install the CUDA Toolkit or set PATH to include it."; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found on PATH."; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: python interpreter '$PYTHON' not found. Set PYTHON=<interpreter>."; exit 1; }
nvcc --version | tail -1
"$PYTHON" -c 'import sys; v=sys.version_info; assert v >= (3,10), f"Python >= 3.10 required, found {v.major}.{v.minor}"; print(f"python {v.major}.{v.minor}.{v.micro}")' || {
  echo "ERROR: this project requires Python >= 3.10. Use a newer interpreter via PYTHON=<path> (e.g. PYTHON=python3.11)."
  exit 1
}
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found. The build can proceed, but running the kernels requires an NVIDIA GPU + driver."
fi

echo "==> [2/7] Installing Python package (editable)"
if [ "${SKIP_PIP_INSTALL:-0}" = "1" ]; then
  echo "    SKIP_PIP_INSTALL=1 set; skipping editable install."
else
  "$PYTHON" -m pip install --upgrade pip >/dev/null
  "$PYTHON" -m pip install -e ".[dev]"
fi

BIN="$BUILD_DIR/gpu_benchmark"
[ -f "$BIN" ] || BIN="$BUILD_DIR/Release/gpu_benchmark.exe"
[ -f "$BIN" ] || BIN="$BUILD_DIR/gpu_benchmark.exe"

echo "==> [3/7] Building CUDA harness"
cmake -S benchmarks -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --config Release

BIN="$BUILD_DIR/gpu_benchmark"
[ -f "$BIN" ] || BIN="$BUILD_DIR/Release/gpu_benchmark.exe"
[ -f "$BIN" ] || BIN="$BUILD_DIR/gpu_benchmark.exe"
[ -f "$BIN" ] || { echo "ERROR: built binary not found under $BUILD_DIR"; exit 1; }
echo "    binary: $BIN"

echo "==> [4/7] Smoke-testing new kernels directly (with --verify)"
for KERNEL in memcpy_bandwidth stencil_1d; do
  echo "    -- $KERNEL"
  "$BIN" --kernel "$KERNEL" --problem-size 1048576 --block-size 256 --warmups 5 --repeats 10 --verify
done

echo "==> [5/7] Running full benchmark sweep"
"$PYTHON" -m gpu_kernel_analyzer benchmark sweep \
  --binary "$BIN" \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir "$OUT_DIR"

echo "==> [6/7] Validating artifacts and generating analysis/report"
"$PYTHON" -m gpu_kernel_analyzer artifacts validate --run-dir "$OUT_DIR"
ANALYZE_ARGS=(--run-dir "$OUT_DIR")
if [ -n "${PEAK_GFLOPS:-}" ]; then ANALYZE_ARGS+=(--peak-gflops "$PEAK_GFLOPS"); fi
if [ -n "${PEAK_BW_GBPS:-}" ]; then ANALYZE_ARGS+=(--peak-bandwidth-gbps "$PEAK_BW_GBPS"); fi
"$PYTHON" -m gpu_kernel_analyzer analyze full "${ANALYZE_ARGS[@]}"

echo "==> [7/7] Running Python test suite"
"$PYTHON" -m pytest -q

echo
echo "==> DONE. Validation artifacts written to: $OUT_DIR"
echo "    - $OUT_DIR/benchmark_summary.csv"
echo "    - $OUT_DIR/analysis_speedup.csv"
echo "    - $OUT_DIR/REPORT.md"
echo "    - $OUT_DIR/plots/"
