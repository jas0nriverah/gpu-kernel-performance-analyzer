#!/usr/bin/env bash
# Capture the same baseline, repeated scaling suite, and edge checks on any CUDA GPU.
# Example: PYTHON=.venv/bin/python OUT_DIR=outputs/h100 bash scripts/cross_validate_gpu.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-build}"
OUT_DIR="${OUT_DIR:-outputs/cross_validation}"
if [[ -e "$OUT_DIR" ]]; then
  echo "Choose a new OUT_DIR; refusing to overwrite $OUT_DIR" >&2
  exit 1
fi
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${CUDA_ARCHITECTURES:-}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
fi
cmake -S benchmarks -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --config Release -j 4
mkdir -p "$OUT_DIR"
nvcc --version > "$OUT_DIR/nvcc-version.txt"
"$PYTHON" -m pip freeze > "$OUT_DIR/requirements-lock.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.limit --format=csv > "$OUT_DIR/gpu-before.csv"
for suite in baseline trial-1 trial-2 trial-3 edges; do
  config=configs/cross_validation.yaml
  if [[ "$suite" == baseline ]]; then config=configs/benchmark_scenarios.yaml; fi
  if [[ "$suite" == edges ]]; then config=configs/correctness_edges.json; fi
  "$PYTHON" -m gpu_kernel_analyzer benchmark sweep --binary "$BUILD_DIR/gpu_benchmark" \
    --scenarios "$config" --outdir "$OUT_DIR/$suite"
  "$PYTHON" -m gpu_kernel_analyzer artifacts validate --run-dir "$OUT_DIR/$suite"
done
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.limit --format=csv > "$OUT_DIR/gpu-after.csv"
"$PYTHON" scripts/summarize_results.py --results "$OUT_DIR"
echo "Capture complete: $OUT_DIR. Run compare against the baseline GPU in a separate directory."
