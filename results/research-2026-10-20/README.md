# Research figures — measured evidence and model limits

Rebuild from repository inputs with `.venv/bin/python scripts/plot_research_results.py`. No files under `outputs/`, executable model bundles, model fitting, or network data are used. Matplotlib renders PNG and editable SVG with the Agg backend. The immutable quality evidence remains unchanged.

`figure-manifest.json` records generator/environment versions, every input SHA-256, calculation checks, figure hashes and exported plot-data hashes. CSV files under `data/` expose the calculations. Regeneration may change SVG metadata timestamps; input/calculation reproducibility is distinct from byte-identical rendering.

## Capture coverage

![01_capture_coverage](01_capture_coverage.png)

Captured observations and retained steady rows, with development and external/prospective roles explicitly separated. Each capture contains 24 verified workload windows across three trials.

[SVG](01_capture_coverage.svg) · [PNG](01_capture_coverage.png)

## Matched timing scenarios

![02_matched_timing](02_matched_timing.png)

Each dot is one scenario's median runtime ratio across three corresponding trial pairs; vertical segments span those three ratios. All 74 scenario identities match for every device and trial. The dashed line is H100 parity.

[SVG](02_matched_timing.svg) · [PNG](02_matched_timing.png)

## Power and energy tradeoffs

![03_power_energy_tradeoffs](03_power_energy_tradeoffs.png)

Each point represents a device/configuration; points and ranges are the median and full range of three trial summaries. Estimated joules per launch equal steady mean sensor watts divided by whole-window launch rate. Runtime includes host launch and synchronization overhead.

[SVG](03_power_energy_tradeoffs.svg) · [PNG](03_power_energy_tradeoffs.png)

## Model selection

![04_model_selection](04_model_selection.png)

Bars are unweighted mean fold RMSE; dots show every held-out fold. A workload mean wins the two-device objective. A forest wins the separate known-device objective. Both selections were frozen before their subsequent external/prospective tests.

[SVG](04_model_selection.svg) · [PNG](04_model_selection.png)

## Prospective H200 validation

![05_prospective_h200](05_prospective_h200.png)

A frozen forest evaluated once on a later H200 capture: calibration, a logarithmic-count residual histogram, and per-workload signed-error distributions. All 1,153 rows are retained, including 24 errors above 25 W. Box whiskers span the full observed minimum/maximum; boxes span quartiles.

[SVG](05_prospective_h200.svg) · [PNG](05_prospective_h200.png)

## External-device results

![06_external_device_failure](06_external_device_failure.png)

Final RMSE on GH200 and Spark with separate axis ranges. The historical fixed H100 linear reference is shown retrospectively, not promoted to a new winner. The known-device forest is challenged off device without refitting; its low within-device score does not transfer.

[SVG](06_external_device_failure.svg) · [PNG](06_external_device_failure.png)

## Feature domains

![07_feature_domains](07_feature_domains.png)

Observed telemetry feature distributions by capture. Boxes span quartiles; whiskers include the full minimum/maximum. Static memory clocks distinguish the development GPUs. Spark's unsupported memory-clock sensor remains missing, not zero.

[SVG](07_feature_domains.svg) · [PNG](07_feature_domains.png)

## Interpretation limits

Device timing comparisons differ in architecture, host CPU, compiler/toolkit and capture date. Trial ranges and correlated rows are not confidence intervals or independent sample counts. Estimated energy combines steady sensor mean power with whole-window launch rate; it is not isolated per-kernel energy. Spark's sensor is GPU-only and excludes CPU/whole-system power. Development selection, known-device prospective validation and untouched external-device challenges answer different questions. Failures and full residual tails are retained. A100 is queued and contributes no completed observation to these figures. Fitted-model feature importance is omitted because no independently archived importance vector is available and weights are deliberately not required for reproduction.
