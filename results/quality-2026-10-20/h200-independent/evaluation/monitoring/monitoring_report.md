# Monitoring Report

## Summary
- Status: `ok`
- Reference rows: `2276`
- Input file: `outputs/quality-2026-10-20/h200-independent/power/steady_telemetry.csv`
- Warning count: `0`

## Warnings
- No warnings.

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
| graphics_clock_mhz | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| graphics_clock_mhz | numeric | mean_shift_std | ok | Input mean is close to reference. | 1980.0 | 1980.0 | 2.0 |
| graphics_clock_mhz | numeric | outside_reference_range_share | ok | Values are mostly within the reference min/max range. | [1980.0, 1980.0] | 0.0 | 0.2 |
| memory_clock_mhz | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| memory_clock_mhz | numeric | mean_shift_std | ok | Input mean is close to reference. | 2913.835676625659 | 3201.0 | 2.0 |
| memory_clock_mhz | numeric | outside_reference_range_share | ok | Values are mostly within the reference min/max range. | [2619.0, 3201.0] | 0.0 | 0.2 |
| memory_utilization_pct | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
