# Native H100 / H200 / GH200 validation

**GH200 was validated on 2026-10-20.** Native smoke, full timing, and sustained power capture completed on `hopper1.cc.gatech.edu` using an existing user allocation on the Planet Express cluster. For future runs, obtain the cluster operator's official login/portal and allocation instructions; execute inside an allocated GPU node.

GH200 combines an Arm Grace CPU and Hopper GPU ([NVIDIA architecture description](https://docs.nvidia.com/dccpu/index.html)). An H200 result from an x86_64 host does not validate GH200. Build locally on each allocated machine; create a new native Python environment on GH200 (`aarch64`) and install project dependencies there using the site's supported Python/CUDA toolchain. Do not copy an x86_64 virtual environment, compiled wheels, build directory, or benchmark executable. The workflow never installs dependencies or changes clocks, power limits, modules, or system settings.

After loading the site's CUDA toolkit (Hopper-capable nvcc and CMake >=3.21) and preparing a native Python >=3.10 environment with this project installed, run:

```bash
# Use the exact allocated full-GPU UUID reported by nvidia-smi.
GPU_UUID=GPU-<allocated-uuid> PYTHON=/path/to/native/venv/bin/python \
  MODE=smoke bash scripts/cross_validate_gpu.sh

GPU_UUID=GPU-<allocated-uuid> PYTHON=/path/to/native/venv/bin/python \
  MODE=full bash scripts/cross_validate_gpu.sh
```

`MODE=smoke` runs the boundary correctness suite and validates its artifacts; it is not performance evidence. `MODE=full` (the default, preserving earlier usage) runs the baseline, three scaling trials, edge checks, artifact validation, and timing summary. Run smoke first, then full serially on an idle, exclusively allocated device. Do not overlap power capture, another benchmark, or another user's GPU workload. The preflight is an initial check, not a scheduler lock; exclusive allocation must prevent later interference.

`PYTHON`, `OUT_DIR`, and `BUILD_DIR` remain configurable. The default output is `outputs/<GPU-UUID>/<UTC-time>-<pid>/`, with its own fresh build subdirectory. Existing nonempty output/build directories are rejected to prevent cache and binary reuse. `CUDA_ARCHITECTURES` is detected from the selected GPU's compute capability (Hopper is 90); if supplied, it must match. No host library directory is hardcoded. Existing single-device `CUDA_VISIBLE_DEVICES` is respected and checked against `GPU_UUID`; multiple devices, partial UUIDs, MIG, unknown utilization, active compute processes, and conflicting selections fail conservatively. A single physical nvidia-smi index is accepted, then normalized to the exact UUID. The CUDA runtime independently confirms that ordinal zero has the PCI bus address associated with that UUID, and all build/benchmark subprocesses inherit that one-device visibility.

`preflight.json` records host architecture, native Python path/version, GPU UUID/name/driver/memory/clocks/power limit, resolved runtime library and runtime/compiler/CMake versions, git revision/status, tracked dirty diff, and untracked filenames. `requirements-lock.txt`, build/configure logs, final GPU state, and the suite artifacts accompany it. Untracked source contents are not copied: commit intended source or archive those files explicitly for exact reproduction. Avoid placing secrets in source diffs. Preflight and version discovery are read-only; CUDA context initialization may briefly reserve device memory. If a failure leaves partial output, preserve it for diagnosis and choose a fresh output directory for retry.

Power is a separate experiment: first finish the timing workflow, then use the repository's power capture CLI with the same selected UUID/native environment and a fresh output path. Power sample periods must cover enough repeated work to produce measured (not empty) samples; keep power and timing results labeled separately. The timing command above makes no energy claim.

For H100/H200/GH200 comparison, retain all three per-device run directories, compare identical scenario/kernel/problem/block identifiers, review all three timing trials and correctness, and record differences in memory capacity, clocks/power limit, driver/toolkit and host CPU. GH200's coherent Grace memory is not automatically exercised by these device-allocation kernels; this protocol tests the existing kernels, not CPU–GPU coherent-memory performance.

CPU-only guard checks: `python -m pytest -q tests/test_hardware_preflight.py`; shell syntax: `bash -n scripts/cross_validate_gpu.sh`. The GH200 run described below passed native smoke/full execution and artifact integrity checks.


## Completed GH200 run

The test used a fresh native `aarch64` Python 3.12.3 environment, GCC 13.3, CMake 4.4.4, CUDA 12.8.93/runtime 12.8, and driver 570.211.01. The selected device was `NVIDIA GH200 480GB`, UUID `GPU-d0a568f0-708f-e632-a937-78e43236ecab`, with 97,871 MiB GPU memory, compute capability 9.0, and a recorded 900 W power limit. The native CUDA runtime was `/usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so.12.8.90`. These settings were observed, not changed.

The correctness smoke passed 18 scenarios. Full timing passed 24 baseline scenarios, three independent 74-scenario scaling suites, and 18 edge scenarios, with all artifact validation and aggregation checks passing. Separate sustained power capture completed 24 windows (eight scenarios × three trials) using `configs/power_h100.yaml`, 15-second windows, and 2,200 raw telemetry rows. Raw capture hashes and independently recomputed power summaries passed verification. These counts describe validation coverage; final holdout target values and model scores remain outside this protocol document.

Local evidence is under `outputs/quality-2026-10-20/gh200/`: `smoke/`, `timing/`, `power/`, installation/environment logs, `execution-receipt.json`, and `source-manifest.json`. The exact working-tree snapshot was captured at `2026-10-20T20:25:03.622686Z`, based on revision `335069725968e031ed82c1e847df4853eb9c5976`, with SHA256 hashes for 438 source/evidence files. All 48 benchmark/config/analyzer source files checked matched that snapshot. Later serving improvements and documentation updates are separate from the captured benchmark source. The remote installation and raw evidence remain in `/nethome/jriverah3/gpu-kernel-quality-20261020`; the pre-existing allocation was preserved.

A fresh native environment was created with `python3 -m venv .venv-gh200`, followed by explicit `pip install -e ".[dev]" cmake` inside the allocation. This preparation is separate from the validation script, which installs nothing. The native environment is excluded through remote `.git/info/exclude`; no x86_64 environment or binary was reused. Downloaded build executables are Arm artifacts and must not be executed on the H200 host.

Power targets are GPU-board readings; they do not measure Grace CPU, whole-superchip, or full-server energy. Timing comparisons also differ in host CPU, driver/toolkit versions, and capture dates, so they cannot isolate hardware as the sole cause of differences. The native preflight guard tests passed on GH200; the subsequent local bounded-probe improvement passed 13 CPU guard tests and does not change kernel or capture protocol.
