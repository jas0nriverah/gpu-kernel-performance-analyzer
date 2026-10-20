# Measured power warehouse

This standard-library SQLite module turns completed measured GPU board-power captures into queryable, reproducible observations. It does not require pandas or a database server. Raw files remain unchanged.

```bash
python -m gpu_power_pipeline.warehouse ingest --database /tmp/power.sqlite \
  results/h100-power-2026-09-22 \
  outputs/prizepicks-h200-2026-10-20/power
python -m gpu_power_pipeline.warehouse query --database /tmp/power.sqlite
```

`ingest_capture(database, directory)` returns `True` when inserted and `False` for an identical existing run. A repeated run ID with different manifest content fails. Each capture commits independently; a multi-directory command can commit earlier captures before a later capture fails. Input validation completes before database writes. A single capture's parent and child rows commit together under `BEGIN IMMEDIATE`; concurrent ingestion of identical captures produces one insertion. SQLite foreign keys are enabled on every ingest connection. External SQL writers must also enable `PRAGMA foreign_keys=ON`.

Capture schema version 1 and warehouse `PRAGMA user_version=1` are supported. Tables:

| Table | Key | Contents |
| --- | --- | --- |
| `runs` | `run_id` | Device UUID/name, full source commit, original absolute directory, manifest SHA-256 and JSON |
| `artifacts` | `run_id, path` | Every manifest-listed artifact digest plus manifest digest |
| `trials` | `run_id, trial, scenario_id` | Workload and original summary JSON |
| `samples` | `run_id, epoch_s` | Raw row JSON, watts, workload/trial reference and steady membership |

All manifest-listed hashes are verified, including worker artifacts, and paths must remain within the capture directory. The four required artifacts are `telemetry.csv`, `steady_telemetry.csv`, `summary.json`, and `summary.csv`. JSON and CSV summaries must agree. Run, device, session and workload identities must agree across telemetry and manifest. Duplicate sample identities, unknown trials, missing values, non-finite measurements, negative power/clocks/query durations, utilization outside 0–100%, temperatures below absolute zero, unverified workloads and mismatched counts fail ingestion. Optional utilization, clock and temperature sensors may be unavailable: capture emits empty CSV cells, which remain empty in stored raw JSON and are never replaced with zero. Their columns are still required, and present values must pass finite/physical checks; literal `N/A`, whitespace, infinity and NaN are rejected at ingestion. Required power, timing and identity values cannot be missing. Steady rows must match raw samples exactly except for their phase. Trial counts, power extrema, integrated energy, span and mean power are recomputed from steady samples and checked against summaries.

`steady_trial_stats` uses SQL window functions to reproduce trapezoidal power integration from the capture protocol, rather than assuming an arithmetic sample average. `device_workload_stats` combines trials with elapsed-time weighting and exposes sample counts:

```sql
SELECT gpu_name, workload, trial_count, sample_count, mean_power_w
FROM device_workload_stats
ORDER BY gpu_name, workload;
```

To compare a source summary directly:

```sql
SELECT run_id, trial, scenario_id, sample_count, mean_power_w, energy_j, span_s
FROM steady_trial_stats;
```

These are measured board-power observations, not per-kernel energy attribution. No prediction table or cross-device generalization claim is introduced. The warehouse stores the source commit and manifest (including dirty-source metadata); artifact hashes verify supplied capture integrity, not the authenticity of the machine or author. It does not compare historical source hashes to the current checkout. Copying identical capture bytes to another directory is idempotent and retains the first recorded source directory. Formatting changes to a manifest count as conflicting content. There is no automatic schema migration, deduplication across different run IDs, or upper wattage limit because GPU board limits vary. This is a local analytical warehouse, not a multi-tenant service.

Validation: `python -m pytest tests/power/test_warehouse.py -q` covers archived H100 SQL parity, idempotence, hash corruption, duplicate samples, non-finite values, identity conflicts, transaction rollback after an injected insertion failure, and concurrent identical ingestion. Both archived H100 and local H200 captures were ingested and re-ingested successfully during development.
