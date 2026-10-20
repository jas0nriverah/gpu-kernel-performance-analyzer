# Monitoring Report

## Summary
- Status: `warning`
- Reference rows: `2276`
- Input file: `outputs/quality-2026-10-20/spark/power/steady_telemetry.csv`
- Warning count: `1`

## Warnings
- gpu_name: 100.0% of values were not observed in the reference profile.

## Drift Checks
| feature | feature_type | check | severity | message | reference_value | input_value | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| power_watts | extra | extra_feature | info | Input includes extra feature 'power_watts' that the model does not use. |  |  |  |
| timestamp | extra | extra_feature | info | Input includes extra feature 'timestamp' that the model does not use. |  |  |  |
| session_id | extra | extra_feature | info | Input includes extra feature 'session_id' that the model does not use. |  |  |  |
| gpu_utilization_pct | extra | extra_feature | info | Input includes extra feature 'gpu_utilization_pct' that the model does not use. |  |  |  |
| memory_utilization_pct | extra | extra_feature | info | Input includes extra feature 'memory_utilization_pct' that the model does not use. |  |  |  |
| temperature_c | extra | extra_feature | info | Input includes extra feature 'temperature_c' that the model does not use. |  |  |  |
| graphics_clock_mhz | extra | extra_feature | info | Input includes extra feature 'graphics_clock_mhz' that the model does not use. |  |  |  |
| memory_clock_mhz | extra | extra_feature | info | Input includes extra feature 'memory_clock_mhz' that the model does not use. |  |  |  |
| gpu_uuid | extra | extra_feature | info | Input includes extra feature 'gpu_uuid' that the model does not use. |  |  |  |
| run_id | extra | extra_feature | info | Input includes extra feature 'run_id' that the model does not use. |  |  |  |
| scenario_id | extra | extra_feature | info | Input includes extra feature 'scenario_id' that the model does not use. |  |  |  |
| trial | extra | extra_feature | info | Input includes extra feature 'trial' that the model does not use. |  |  |  |
| backend | extra | extra_feature | info | Input includes extra feature 'backend' that the model does not use. |  |  |  |
| power_scope | extra | extra_feature | info | Input includes extra feature 'power_scope' that the model does not use. |  |  |  |
| source_file | extra | extra_feature | info | Input includes extra feature 'source_file' that the model does not use. |  |  |  |
| gpu_name | categorical | missing_rate_delta | ok | Categorical missing rate compared with reference. | 0.0 | 0.0 | 0.2 |
| gpu_name | categorical | unseen_category_share | warning | 100.0% of values were not observed in the reference profile. | 2.0 | 1.0 | 0.2 |
| workload_type | categorical | missing_rate_delta | ok | Categorical missing rate compared with reference. | 0.0 | 0.0 | 0.2 |
| workload_type | categorical | unseen_category_share | ok | Categories are covered by the reference profile or too high-cardinality to flag. | 6.0 | 0.0 | 0.2 |
