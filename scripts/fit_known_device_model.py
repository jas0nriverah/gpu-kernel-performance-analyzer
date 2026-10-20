#!/usr/bin/env python3
"""Freeze a known-device, new-session winner from previously fixed trial folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.generalization import (
    CATEGORICAL,
    NUMERIC,
    build_candidate,
    candidate_plan,
    choose_candidate,
    write_frozen,
)
from gpu_power_pipeline.inference import PowerModel
from gpu_power_pipeline.monitoring import build_reference_profile
from gpu_power_pipeline.persistence import ModelMetadata, new_version_id, save_model_bundle


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_known_device(plan, metrics, expected_folds):
    """Score whole-trial folds only, requiring every fixed candidate and fold."""
    diagnostic = metrics.loc[metrics['split'].eq('whole_trial_diagnostic')]
    if set(diagnostic['candidate']) != {candidate['id'] for candidate in plan}:
        raise ValueError('Trial diagnostics must cover exactly the frozen candidates')
    scores = []
    for candidate in plan:
        rows = diagnostic.loc[diagnostic['candidate'].eq(candidate['id'])]
        if len(rows) != len(expected_folds) or set(rows['fold']) != set(expected_folds):
            raise ValueError('Every candidate requires each whole-trial diagnostic exactly once')
        values = rows['rmse_w'].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Trial RMSE must be finite and nonnegative')
        scores.append((float(values.mean()), float(values.max()), candidate))
    mean, worst, winner = min(scores, key=lambda row: row[0])
    return {'candidate': winner, 'mean_trial_rmse_w': mean, 'worst_trial_rmse_w': worst,
            'selection_rule': 'minimum unweighted mean RMSE across six whole-device/trial groups; ties use frozen plan order'}


def fit_known_device(development_dir, outdir):
    development_dir, outdir = Path(development_dir), Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError('Output directory must be new or empty')
    original_plan_path = development_dir / 'plan.json'
    original_selection_path = development_dir / 'selection.json'
    original_fit = json.loads((development_dir / 'fit.json').read_text())
    original_selection = json.loads(original_selection_path.read_text())
    original_plan = json.loads(original_plan_path.read_text())
    if sha256(original_plan_path) != original_selection['plan_sha256']:
        raise ValueError('Original frozen plan hash mismatch')
    if sha256(original_selection_path) != original_fit['selection_sha256']:
        raise ValueError('Original frozen selection hash mismatch')
    if original_plan['candidates'] != candidate_plan():
        raise ValueError('Original grid differs from the fixed candidate implementation')
    import gpu_power_pipeline.generalization as implementation
    for name, digest in original_plan['source_files_sha256'].items():
        if sha256(Path(implementation.__file__).with_name(name)) != digest:
            raise ValueError(f'Development source hash mismatch: {name}')
    metrics_path = development_dir / 'fold_metrics.csv'
    metrics = pd.read_csv(metrics_path)
    replay = choose_candidate(original_plan['candidates'], metrics.to_dict('records'))
    if replay['candidate'] != original_selection['candidate']:
        raise ValueError('Device scores do not reproduce original frozen winner')
    frames = []
    for item in original_plan['inputs']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Development input hash mismatch')
        frame = load_dataset('local_nvidia_smi', data_path=item['path'])
        if len(frame) != len(pd.read_csv(item['path'])) or not np.isfinite(frame['power_watts']).all():
            raise ValueError('Every development input row must have a finite target')
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != original_fit['training_rows'] or sorted(frame['gpu_uuid'].unique()) != original_fit['training_gpu_uuids']:
        raise ValueError('Development identities or row counts changed')
    groups = frame['gpu_uuid'].astype(str) + ':trial:' + frame['trial'].astype(str)
    expected_folds = sorted(groups.unique())
    if len(expected_folds) != 6:
        raise ValueError('Expected exactly six fixed whole-trial groups')
    selection = select_known_device(original_plan['candidates'], metrics, expected_folds)
    for fold in expected_folds:
        rows = metrics.loc[metrics['split'].eq('whole_trial_diagnostic') & metrics['fold'].eq(fold)]
        count = int(groups.eq(fold).sum())
        if not rows['test_rows'].eq(count).all() or not rows['train_rows'].eq(len(frame)-count).all():
            raise ValueError('Trial diagnostic row counts differ from development inputs')
    outdir.mkdir(parents=True, exist_ok=True)
    configuration = dict(original_plan)
    configuration.update(created_utc=datetime.now(timezone.utc).isoformat(), objective='known_device_new_session',
                         selection='mean six whole-trial validation RMSE; future independent capture not yet accessed',
                         parent_plan_sha256=sha256(original_plan_path),
                         parent_selection_sha256=sha256(original_selection_path),
                         diagnostic_input={'path': str(metrics_path.resolve()), 'sha256': sha256(metrics_path)},
                         selection_source_sha256=sha256(__file__))
    write_frozen(outdir / 'plan.json', configuration)
    selection.update(frozen_utc=datetime.now(timezone.utc).isoformat(), objective='known_device_new_session',
                     plan_sha256=sha256(outdir / 'plan.json'),
                     scope='Known H100/H200 GPUs, future session/capture. Whole-trial scores are development validation, not independent test evidence.')
    write_frozen(outdir / 'selection.json', selection)
    candidate = selection['candidate']
    features = candidate['features']
    numeric = [f for f in features if f in NUMERIC]
    categorical = [f for f in features if f in CATEGORICAL]
    with threadpool_limits(limits=1):
        pipeline = build_candidate(candidate)
        pipeline.fit(frame[features], frame['power_watts'])
        metadata = ModelMetadata(model_name=candidate['id'], target='power_watts', numeric_features=numeric,
                                 categorical_features=categorical, feature_order=features, metrics={},
                                 split_strategy='known_device_whole_trial_validation_then_refit', source='local_nvidia_smi',
                                 version=new_version_id(), reference_profile=build_reference_profile(
                                     frame, list(dict.fromkeys(features + ['gpu_name'])), numeric,
                                     list(dict.fromkeys(categorical + ['gpu_name']))))
        bundle = save_model_bundle(pipeline, metadata, outdir / 'registry')
        loaded = PowerModel.from_dir(bundle)
        np.testing.assert_allclose(loaded.predict(frame), pipeline.predict(frame[features]), rtol=1e-12, atol=1e-12)
    result = {'selection_sha256': sha256(outdir / 'selection.json'), 'saved_model': str(bundle),
              'saved_model_sha256': sha256(bundle / 'model.joblib'), 'training_rows': len(frame),
              'training_gpu_uuids': sorted(frame['gpu_uuid'].unique().tolist()), 'selected_candidate': candidate['id'],
              'saved_reload_parity': True, 'objective': 'known_device_new_session',
              'frozen_utc': selection['frozen_utc']}
    write_frozen(outdir / 'fit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-dir', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fit_known_device(args.development_dir, args.outdir), indent=2))


if __name__ == '__main__':
    main()
