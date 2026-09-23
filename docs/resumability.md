# Resumable benchmark sweeps

Run a sweep, then repeat the same command with `--resume` after an interruption:

```bash
python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir artifacts/vector-add

python -m gpu_kernel_analyzer benchmark sweep \
  --binary build/gpu_benchmark \
  --scenarios configs/benchmark_scenarios.yaml \
  --outdir artifacts/vector-add \
  --resume
```

`benchmark sweep --resume` continues a sweep interrupted in the same output location. An interrupted run stores its checkpoint and validated per-scenario binary output in a hidden sibling directory named `.<outdir>.partial`; the requested output directory is created only after all scenarios finish and the generated artifacts pass normal validation. The complete directory is published with one same-filesystem rename, so readers see either no result directory or the complete result directory.

The checkpoint binds the binary path and SHA-256, scenario file path and SHA-256, expanded scenario list, interpreter, timeout, and benchmark JSON protocol version. Each completed scenario is stored atomically with its own digest. A restart refuses identity mismatches, malformed checkpoint data, checksum mismatches, duplicate or out-of-range scenarios, and inconsistent device metadata. The checksums detect ordinary edits and corruption; they do not provide authenticity against someone who can rewrite the checkpoint and recompute its hashes. Existing nonempty output directories remain protected; an interrupted partial directory requires `--resume` and is never silently replaced.

A sibling advisory file lock uses POSIX `flock`, which the operating system releases after process exit or `SIGKILL`. Concurrent runs are serialized on local filesystems that implement POSIX advisory locks; resumable sweeps require a POSIX platform. Keep the partial and output directories on the same filesystem so final publication remains a single atomic rename. Failures before publication leave the partial checkpoint available for another resume attempt. After publication, the partial directory is removed. This protects against process interruption; it does not promise durability against sudden power loss.

The worker reports a CUDA device UUID. On restart, the analyzer invokes the same binary with bounded `--device-info` and compares the live selected-device UUID with the checkpoint before reusing measurements. Runs with device metadata but no UUID cannot resume.
