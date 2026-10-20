# Monitoring Report

## Summary
- Status: `warning`
- Reference rows: `1123`
- Input file: `outputs/quality-2026-10-20/gh200/power/steady_telemetry.csv`
- Warning count: `1`

## Warnings
- gpu_name: 100.0% of values were not observed in the reference profile.

## Drift Checks
| feature | feature_type | check | severity | message | reference_value | input_value | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| power_watts | extra | extra_feature | info | Input includes extra feature 'power_watts' that the model does not use. |  |  |  |
| timestamp | extra | extra_feature | info | Input includes extra feature 'timestamp' that the model does not use. |  |  |  |
| session_id | extra | extra_feature | info | Input includes extra feature 'session_id' that the model does not use. |  |  |  |
| gpu_uuid | extra | extra_feature | info | Input includes extra feature 'gpu_uuid' that the model does not use. |  |  |  |
| run_id | extra | extra_feature | info | Input includes extra feature 'run_id' that the model does not use. |  |  |  |
| scenario_id | extra | extra_feature | info | Input includes extra feature 'scenario_id' that the model does not use. |  |  |  |
| trial | extra | extra_feature | info | Input includes extra feature 'trial' that the model does not use. |  |  |  |
| backend | extra | extra_feature | info | Input includes extra feature 'backend' that the model does not use. |  |  |  |
| power_scope | extra | extra_feature | info | Input includes extra feature 'power_scope' that the model does not use. |  |  |  |
| source_file | extra | extra_feature | info | Input includes extra feature 'source_file' that the model does not use. |  |  |  |
| gpu_utilization_pct | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| gpu_utilization_pct | numeric | mean_shift_std | ok | Input mean is close to reference. | 100.0 | 100.0 | 2.0 |
| gpu_utilization_pct | numeric | outside_reference_range_share | ok | Values are mostly within the reference min/max range. | [100.0, 100.0] | 0.0 | 0.2 |
| memory_utilization_pct | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| memory_utilization_pct | numeric | mean_shift_std | ok | Input mean is close to reference. | 62.443455031166515 | 56.74858528698464 | 2.0 |
| memory_utilization_pct | numeric | outside_reference_range_share | ok | Values are mostly within the reference min/max range. | [0.0, 100.0] | 0.0 | 0.2 |
| temperature_c | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| temperature_c | numeric | mean_shift_std | ok | Input mean is close to reference. | 46.80409617097062 | 49.984640258690376 | 2.0 |
| temperature_c | numeric | outside_reference_range_share | ok | Values are mostly within the reference min/max range. | [39.0, 53.0] | 0.07518189167340339 | 0.2 |
| graphics_clock_mhz | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
