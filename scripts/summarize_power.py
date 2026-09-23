#!/usr/bin/env python3
"""Regenerate a GPU power report from captured, hashed telemetry and summaries."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gpu_kernel_analyzer.artifacts import sha256_file
from gpu_kernel_analyzer.power import summarize_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Cannot publish an incomplete capture')
    for relative, digest in manifest['artifacts_sha256'].items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f'Capture artifact changed: {relative}')
    samples = list(csv.DictReader((root / 'telemetry.csv').open()))
    for row in samples:
        for key in ('epoch_s', 'monotonic_s', 'power_watts', 'query_duration_s'):
            row[key] = float(row[key])
        if row['gpu_uuid'] != manifest['device']['gpu_uuid']:
            raise ValueError('Telemetry UUID does not match capture device')
    summaries = json.loads((root / 'summary.json').read_text())
    groups = {}
    trim = manifest['protocol']['trim_seconds']
    for row in summaries:
        window = row['window']
        recomputed = summarize_power(samples, window['start_epoch_s'] + trim, window['end_epoch_s'] - trim)
        for key, value in recomputed.items():
            if abs(row[key] - value) > 1e-8:
                raise ValueError(f'Summary disagrees with raw telemetry: {key}')
        groups.setdefault(row['scenario_id'], []).append(row)
    aggregate = []
    for key, rows in sorted(groups.items()):
        if len(rows) != manifest['protocol']['trials']:
            raise ValueError(f'Missing trials: {key}')
        aggregate.append(dict(scenario_id=key, kernel=rows[0]['kernel'], problem_size=rows[0]['problem_size'],
                              block_size=rows[0]['block_size'], trials=len(rows),
                              mean_power_w=statistics.mean(r['mean_power_w'] for r in rows),
                              min_trial_power_w=min(r['mean_power_w'] for r in rows),
                              max_trial_power_w=max(r['mean_power_w'] for r in rows),
                              mean_idle_power_w=statistics.mean(r['idle_mean_power_w'] for r in rows),
                              estimated_j_per_launch=statistics.mean(r['estimated_j_per_launch'] for r in rows),
                              effective_gflops=statistics.mean(r['effective_gflops'] for r in rows),
                              effective_gbps=statistics.mean(r['effective_gbps'] for r in rows)))
    with (root / 'aggregate.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    labels = [r['kernel'].replace('_', ' ') + f" / {r['block_size']}" for r in aggregate]
    watts = [r['mean_power_w'] for r in aggregate]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    ax.barh(labels, watts, color=['#2563eb' if r['kernel'].startswith('gemm') else '#059669' for r in aggregate],
            xerr=[[w-r['min_trial_power_w'] for w, r in zip(watts, aggregate, strict=True)],
                  [r['max_trial_power_w']-w for w, r in zip(watts, aggregate, strict=True)]], capsize=4)
    ax.set_xlabel('Measured GPU board power (W); error bars span trial means')
    ax.set_title(f"{manifest['device']['gpu_name']} · {manifest['protocol']['trials']} randomized trials")
    ax.grid(axis='x', alpha=.2)
    fig.savefig(root / 'power.png', dpi=150)
    plt.close(fig)
    lines = ['# Measured power and sustained throughput', '',
             f"Device: **{manifest['device']['gpu_name']}**. {len(summaries)} workload windows; "
             f"{manifest['telemetry_samples']} raw telemetry rows; {manifest['steady_samples']} steady-window rows.", '',
             '![Measured board power](power.png)', '',
             '| Workload | Block | Board power, mean [trial range] W | Sustained effective GB/s | Sustained GFLOP/s | Estimated mJ/launch |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregate:
        lines.append(f"| {r['kernel']} | {r['block_size']} | {r['mean_power_w']:.1f} [{r['min_trial_power_w']:.1f}, {r['max_trial_power_w']:.1f}] | {r['effective_gbps']:.1f} | {r['effective_gflops']:.1f} | {1000*r['estimated_j_per_launch']:.3f} |")
    gemm = {r['kernel']: r for r in aggregate if r['kernel'] in ('gemm_naive', 'gemm_tiled')}
    if len(gemm) == 2 and gemm['gemm_naive']['problem_size'] == gemm['gemm_tiled']['problem_size']:
        naive, tiled = gemm['gemm_naive'], gemm['gemm_tiled']
        speedup = tiled['effective_gflops'] / naive['effective_gflops']
        energy_reduction = 100 * (1 - tiled['estimated_j_per_launch'] / naive['estimated_j_per_launch'])
        lines += ['', '## Optimization result', '',
                  f"At the matched {naive['problem_size']} × {naive['problem_size']} FP32 GEMM size, tiled GEMM achieved **{speedup:.2f}× sustained throughput** and **{energy_reduction:.1f}% lower estimated energy per launch** than naive GEMM. This compares these custom kernels on this device under the captured protocol."]
    lines += ['', '## Measurement scope', '',
              'These are real `nvidia-smi:power.draw` readings, not model predictions. Energy is estimated by trapezoidal integration over covered steady samples. Every workload passed its deterministic correctness check.', '',
              f"Each workload executes batches of 100 launches in a persistent allocation for at least {manifest['protocol']['duration_seconds']:g} seconds. The first and last {manifest['protocol']['trim_seconds']:g} seconds are excluded from power summaries. Launch throughput uses the complete sustained window including batch synchronization and host launch overhead. Estimated energy per launch combines steady mean watts with whole-window launch rate; it is not an isolated per-kernel energy measurement. Effective bytes/FLOPs use the existing workload formulas, not hardware counters.", '',
              'H100 power telemetry is approximately one-second averaged. Samples taken more often are correlated. Trial ranges are descriptive, not confidence intervals. Pre-workload idle readings include recent thermal history; no idle subtraction is applied. No clocks or power limits were changed. GPU-board power is not wall-socket or full-server power.', '',
              'Use same-workload, same-size comparisons for optimization. GEMM bytes count matrix sizes and do not measure actual DRAM traffic. These custom FP32 kernels do not use Tensor Cores or cuBLAS.', '',
              '## Evidence and regeneration', '',
              '- [Manifest, device, binary/source hashes and protocol](manifest.json)',
              '- [Every telemetry query](telemetry.csv)',
              '- [Steady telemetry for modeling](steady_telemetry.csv)',
              '- [Per-trial summaries](summary.csv)',
              '- [Aggregates](aggregate.csv)',
              '- [Worker JSON and measurement boundaries](workers/)', '',
              'From the repository root: `python scripts/summarize_power.py --run-dir results/h100-power-2026-09-22`.', '',
              'Regeneration verifies artifact hashes and recomputes power/energy summaries from raw samples before writing the table and chart.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Validated {len(summaries)} windows and regenerated {root / "REPORT.md"}')


if __name__ == '__main__':
    main()
