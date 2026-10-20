"""Guard the distinction between device selection and known-device trial validation."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    'fit_known_device_model', Path(__file__).resolve().parents[2] / 'scripts/fit_known_device_model.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def diagnostic_scores():
    return pd.DataFrame([
        {'candidate': candidate, 'split': split, 'fold': fold, 'rmse_w': score}
        for candidate, split, score in [('tree', 'whole_trial_diagnostic', 3.0),
                                       ('baseline', 'whole_trial_diagnostic', 20.0),
                                       ('tree', 'device', 80.0), ('baseline', 'device', 10.0)]
        for fold in ['a', 'b', 'c', 'd', 'e', 'f']])


def test_known_device_objective_uses_trial_scores_and_fixed_tie_order():
    plan = [{'id': 'tree'}, {'id': 'baseline'}]
    scores = diagnostic_scores()
    selected = MODULE.select_known_device(plan, scores, ['a', 'b', 'c', 'd', 'e', 'f'])
    assert selected['candidate']['id'] == 'tree'
    assert selected['mean_trial_rmse_w'] == 3
    scores.loc[scores['split'].eq('whole_trial_diagnostic'), 'rmse_w'] = 3
    assert MODULE.select_known_device(plan, scores, list('abcdef'))['candidate']['id'] == 'tree'


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'nonfinite', 'unknown'])
def test_diagnostic_contract_rejects_incomplete_or_invalid_metrics(failure):
    scores = diagnostic_scores()
    if failure == 'missing':
        scores = scores.drop(index=0)
    elif failure == 'duplicate':
        scores = pd.concat([scores, scores.iloc[:1]], ignore_index=True)
    elif failure == 'nonfinite':
        scores.loc[0, 'rmse_w'] = float('nan')
    else:
        scores.loc[0, 'candidate'] = 'unplanned'
    with pytest.raises(ValueError):
        MODULE.select_known_device([{'id': 'tree'}, {'id': 'baseline'}], scores, list('abcdef'))
