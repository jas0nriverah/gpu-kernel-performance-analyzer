#!/usr/bin/env python3
"""Plot final known-device validation without suppressing the residual tail."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.evaluation_dir
    evidence = json.loads((directory / 'evaluation.json').read_text())
    rows = pd.read_csv(directory / 'predictions.csv')
    name = evidence['selected_candidate']
    actual = rows['power_watts'].to_numpy()
    residual = rows[name + '_predicted_w'].to_numpy() - actual
    if not np.isfinite(residual).all() or len(rows) != evidence['test_rows']:
        raise ValueError('Plot requires every finite final-evaluation residual')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    keys = [name, 'pooled_training_workload_mean', 'pooled_training_mean']
    values = [evidence['scores'][key]['rmse_w'] for key in keys]
    bars = axes[0].bar(['Frozen forest', 'Workload mean', 'Global mean'], values,
                       color=['#277DA1', '#90BE6D', '#B8B8B8'])
    axes[0].bar_label(bars, labels=[f'{value:.2f}' for value in values], padding=3)
    axes[0].set(ylabel='RMSE (W)', ylim=(0, max(values)*1.17), title='Independent H200 session')
    for workload, indices in rows.groupby('workload_type', sort=True).indices.items():
        axes[1].scatter(indices, residual[indices], s=9, alpha=.65, label=workload, rasterized=True)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(xlabel='Capture row (all retained)', ylabel='Prediction − observed power (W)',
                title=f'All {len(rows):,} residuals; no trimming')
    axes[1].legend(fontsize=7, loc='upper right')
    fig.suptitle('Frozen known-device random forest: final capture after selection', fontsize=11)
    fig.savefig(directory / 'validation.png', dpi=180)
    fig.savefig(directory / 'validation.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
