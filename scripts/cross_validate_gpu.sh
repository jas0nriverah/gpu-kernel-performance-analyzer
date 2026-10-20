#!/usr/bin/env bash
# Native, single allocated GPU capture. See docs/gh200-validation.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
MODE="${MODE:-full}"
[[ "$MODE" == full || "$MODE" == smoke ]] || { echo 'MODE must be full or smoke' >&2; exit 1; }
PREFLIGHT=$(mktemp)
trap 'rm -f "$PREFLIGHT"' EXIT
"$PYTHON" scripts/hardware_preflight.py --out "$PREFLIGHT"
GPU_UUID=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu"]["uuid"])' "$PREFLIGHT")
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
ARCH=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cuda_architectures"])' "$PREFLIGHT")
OUT_DIR="${OUT_DIR:-outputs/${GPU_UUID}/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
BUILD_DIR="${BUILD_DIR:-$OUT_DIR/build}"
# Never reuse an earlier host's binaries or CMake cache, including empty-looking caches.
for directory in "$OUT_DIR" "$BUILD_DIR"; do
  if [[ -e "$directory" ]] && { [[ ! -d "$directory" ]] || [[ -n "$(ls -A "$directory")" ]]; }; then
    echo "Choose an empty or new directory; refusing to reuse $directory" >&2; exit 1
  fi
done
mkdir -p "$OUT_DIR"
cp "$PREFLIGHT" "$OUT_DIR/preflight.json"
"$PYTHON" -m pip freeze > "$OUT_DIR/requirements-lock.txt"
cmake -S benchmarks -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" 2>&1 | tee "$OUT_DIR/configure.log"
cmake --build "$BUILD_DIR" --config Release -j 4 2>&1 | tee "$OUT_DIR/build.log"
if [[ "$MODE" == smoke ]]; then
  SUITES=(edges)
else
  SUITES=(baseline trial-1 trial-2 trial-3 edges)
fi
for suite in "${SUITES[@]}"; do
  config=configs/cross_validation.yaml
  if [[ "$suite" == baseline ]]; then config=configs/benchmark_scenarios.yaml; fi
  if [[ "$suite" == edges ]]; then config=configs/correctness_edges.json; fi
  "$PYTHON" -m gpu_kernel_analyzer benchmark sweep --binary "$BUILD_DIR/gpu_benchmark" \
    --scenarios "$config" --outdir "$OUT_DIR/$suite"
  "$PYTHON" -m gpu_kernel_analyzer artifacts validate --run-dir "$OUT_DIR/$suite"
done
nvidia-smi -i "$GPU_UUID" --query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.limit --format=csv > "$OUT_DIR/gpu-after.csv"
if [[ "$MODE" == full ]]; then
  "$PYTHON" scripts/summarize_results.py --results "$OUT_DIR"
fi
echo "Capture complete ($MODE): $OUT_DIR"
