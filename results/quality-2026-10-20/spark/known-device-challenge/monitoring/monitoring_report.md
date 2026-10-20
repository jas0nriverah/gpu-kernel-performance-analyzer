# Monitoring Report

## Summary
- Status: `warning`
- Reference rows: `2276`
- Input file: `outputs/quality-2026-10-20/spark/power/steady_telemetry.csv`
- Warning count: `9`

## Warnings
- gpu_utilization_pct: Input mean shifted by 4000000000.00 reference std devs.
- gpu_utilization_pct: 100.0% of values are outside the reference min/max range.
- graphics_clock_mhz: Input mean shifted by 464071358748.78 reference std devs.
- graphics_clock_mhz: 100.0% of values are outside the reference min/max range.
- memory_clock_mhz: Missing rate increased by 1.000.
- memory_clock_mhz: Feature 'memory_clock_mhz' has no valid numeric values in input.
- temperature_c: Input mean shifted by 5.37 reference std devs.
- temperature_c: 89.0% of values are outside the reference min/max range.
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
| gpu_utilization_pct | numeric | mean_shift_std | warning | Input mean shifted by 4000000000.00 reference std devs. | 100.0 | 96.0 | 2.0 |
| gpu_utilization_pct | numeric | outside_reference_range_share | warning | 100.0% of values are outside the reference min/max range. | [100.0, 100.0] | 1.0 | 0.2 |
| graphics_clock_mhz | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| graphics_clock_mhz | numeric | mean_shift_std | warning | Input mean shifted by 464071358748.78 reference std devs. | 1980.0 | 2444.071358748778 | 2.0 |
| graphics_clock_mhz | numeric | outside_reference_range_share | warning | 100.0% of values are outside the reference min/max range. | [1980.0, 1980.0] | 1.0 | 0.2 |
| memory_clock_mhz | numeric | missing_rate_delta | warning | Missing rate increased by 1.000. | 0.0 | 1.0 | 0.2 |
| memory_clock_mhz | numeric | no_valid_numeric_values | warning | Feature 'memory_clock_mhz' has no valid numeric values in input. | 2276 | 0.0 | 1.0 |
| memory_utilization_pct | numeric | missing_rate_delta | ok | Missing rate is close to reference. | 0.0 | 0.0 | 0.2 |
| memory_utilization_pct | numeric | mean_shift_std | ok | Input mean is close to reference. | 56.20694200351494 | 0.0 | 2.0 |
