# Native GPU quality evidence — 2026-10-20

This directory curates H200, native Arm GH200, and native Arm DGX Spark captures; frozen model development and independent evaluation; HTTP serving checks; and relational integrity evidence. The existing H100 evidence remains in sibling result archives.

| Directory | Evidence |
| --- | --- |
| `h200/timing`, `h200/smoke` | Baseline, three scaling trials, correctness edges, raw samples, manifests and native preflight/build logs. |
| `h200/power`, `h200/transfer` | Complete original power capture and the fixed H100-to-H200 transfer experiment. |
| `h200-independent` | A later H200 session collected after model selection froze, with raw capture, execution receipt, independent evaluation and API parity evidence. |
| `gh200/timing`, `gh200/smoke`, `gh200/power` | Native GH200 suites and complete power evidence; GH200 targets were reserved during model selection. |
| `gh200/comparison` | Nine device-pair/trial timing comparisons, their summary and interpretation. |
| `spark/timing`, `spark/smoke`, `spark/power` | Native GB10 `sm_121` suites, 24 sustained power windows, raw telemetry and artifact manifests. |
| `gh200/frozen-evaluation`, `gh200/known-device-challenge`, `spark/frozen-evaluation`, `spark/known-device-challenge` | Frozen-model external tests, including failed transfers and raw predictions; no test-driven selection changes. |
| `a100-queued` | Submitted A100 GPU job 6426654 and dependent CPU evaluation job 6426680, with bounded resources and source amendments. These are queue receipts, not completed A100 results. |
| `spark/comparison` | Three descriptive H200/Spark trial comparisons; these are not same-environment regression gates. |
| `spark` receipts/logs | Native toolchain/environment, source manifest, the two preflight source overrides, and allocation release audit. Only the newly requested Spark allocation was released. |
| `models/model-development` | H100/H200-only device-transfer development, candidate plan, frozen selection, fold predictions and saved-model metadata. |
| `models/known-device-development` | Separate H100/H200 known-device development with whole-trial validation and a frozen selection for a later session. |
| `service` | API verification and localhost HTTP request/latency measurements. |
| `warehouse` | Immutable warehouse count/integrity/idempotency summaries; no mutable database. |
| `validation` | Integrated CPU regression and focused follow-up logs, source hashes and verification receipts. |
| `source` | Original native capture source overlay/hash manifest and later modeling/serving/portability source files. |

Read each experiment's report, selection record and manifest before interpreting results. Development folds, a new device, and a later session answer different questions. Model choices and selection policies were frozen before external evaluation; descriptive telemetry rows and repeated windows are correlated, so row counts are not independent sample counts. Hardware timing comparisons differ in host CPU, driver/toolkit and capture dates; an ineligible comparison gate does not establish a same-host regression. Local HTTP measurements do not establish production capacity.

Spark preserves unsupported memory-clock telemetry as null/empty; no zero substitution is applied. Its preflight separately labels `cudaMemGetInfo` shared/device capacity, rather than fabricating NVML framebuffer memory. Although capture manifests retain the legacy `power_scope=gpu_board` label, GB10 `nvidia-smi` wattage is **GPU-only sensor power**, excluding CPU and total SoC/system power. See [NVIDIA's memory documentation](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html), [official power clarification](https://forums.developer.nvidia.com/t/dgx-spark-power-clarification/349668), and the recorded Spark execution receipt.

`copy-provenance.json` maps each copied artifact to its original local path and SHA256. Most files are byte-for-byte originals. Large out-of-fold CSVs are stored as deterministic gzip: those entries record the uncompressed original SHA256 and byte count plus the stored gzip SHA256. Decompress before checking the original hash or using a CSV reader. Historical absolute paths, manifests and hashes remain unchanged; relocate inputs explicitly when replaying commands.

`SHA256SUMS` hashes every curated file except itself, including the completed external evaluations. Verify the finished archive from this directory with `sha256sum -c SHA256SUMS`. Original capture manifests can reference deliberately omitted executables or original paths; use the curated checksum list for the included files. Build directories, executables, virtual environments, joblib/pickled weights, mutable SQLite databases, lock files and duplicate timing charts are omitted. Model metadata alone is not a loadable model; reconstruct weights from the recorded training inputs, configuration and source.

For the original GH200 source, check out `git_revision` from `source/gh200-quality-source-manifest.json` and extract `source/gh200-quality-source-overlay.tar.gz` over that checkout. Verify all overlay members against the manifest's `files` map. To reproduce Spark, apply `spark/source-overrides/scripts/hardware_preflight.py` and `spark/source-overrides/tests/test_hardware_preflight.py` afterward, then verify against `spark/source-manifest.json`. These are the only two source differences between the capture snapshots; all 48 benchmark/config/analyzer source files match. Later modeling code lives under `source/modeling`; serving, warehouse and reporting code lives under `source/final-quality` with verification receipts recording its hashes. Do not substitute that later code silently for a historical source snapshot.

Build CUDA and install Python dependencies natively on each target host. Never reuse x86_64/aarch64 environments or executables. For new captures, follow `docs/gh200-validation.md` or `docs/dgx-spark-validation.md`, run smoke then full timing on one allocated GPU with fresh output/build directories, and capture power separately afterward. Raw CSV/JSON inspection and warehouse rebuilding require no CUDA. Serving and model reproduction instructions are in `docs/serving.md`, `docs/model-generalization.md` and `docs/warehouse.md`.
